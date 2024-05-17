import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import postprocessing as pp
from torch.utils import data as torch_data

from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.utils import filter_dataset
from miditok.pytorch_data import DatasetMIDI, DatasetJSON, split_files_for_training, DataCollator
from miditok.data_augmentation import augment_dataset

"""Version 6 : This version is absolutely identical to V6. In practice it was just used to differentiate the two versions of the model : the one using the whole dataset (this one) and the one using only midi_classic dataset (that we called V3)"""

class Head(nn.Module):
    def __init__(self, n, n_embedding, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias = False)
        self.query = nn.Linear(n_embedding, head_size, bias = False)
        self.value = nn.Linear(n_embedding, head_size, bias = False)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        self.register_buffer("mask", torch.tril(torch.ones(n,n)))

    def forward(self,x):
        _,T,_ = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        d = K.shape[-1]

        A = Q @ K.transpose(-1,-2)* d**-0.5

        A = A.masked_fill_(self.mask[:T,:T] == 0, float("-inf"))
        A = F.softmax(A, dim=-1)

        Y = A @ V
        
        return Y
    
class MultiHead(nn.Module):
    def __init__(self, n, n_embedding, n_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n, n_embedding, head_size)for _ in range (n_heads)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        y = torch.stack([y_h(x) for y_h in self.heads], dim=0)
        y = torch.sum(y, dim = 0)
        y = self.dropout(y)

        return y
    
class FFN(nn.Module):
    def __init__(self, n_embedding, dropout, factor = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, n_embedding*factor),
            nn.ReLU(),
            nn.Linear(n_embedding*factor, n_embedding),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        y = self.net(x)
        return y
    
class Block(nn.Module):
    def __init__(self, n, n_embedding, n_heads, head_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embedding)
        self.multi_head = MultiHead(n, n_embedding, n_heads, head_size, dropout)
        self.norm2 = nn.LayerNorm(n_embedding)
        self.ffn = FFN(n_embedding, dropout)
    
    def forward(self, x):
        y = self.norm1(x)
        y = self.multi_head(y) + x
        y = self.norm2(y)
        y = self.ffn(y) + y
        return y
    
class Transformer(nn.Module):
    def __init__(self, nbr_tokens, n, n_embedding, n_blocks, n_heads, head_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(nbr_tokens, n_embedding)
        self.position_embedding = nn.Embedding(n, n_embedding)
        self.blocks = nn.ModuleList([Block(n, n_embedding, n_heads, head_size, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(n_embedding, nbr_tokens)
        
    def forward(self, x):
        y = self.embedding(x)  
        y_pos = self.position_embedding(torch.arange(y.shape[1], device = y.device))
        y = y + y_pos
        for block in self.blocks:
            y = block(y)
        y = self.head(y)
        return y
    
class Translater():
    def __init__(self,n):
        TOKENIZER_PARAMS = {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 32,
            "special_tokens": ["PAD", "BOS", "EOS"],
            "use_chords": False,
            "use_rests": False,
            "use_tempos": False,
            "use_time_signatures": False,
            "use_programs": False,
            "num_tempos": 32,
            "tempo_range": (40, 250),
        }
        config = TokenizerConfig(**TOKENIZER_PARAMS)
        self.tokenizer = REMI(config)
        self.n = n
        
    def encode(self, x):
        tokens = self.tokenizer(Path(x))
        return tokens
    
    def get_xy(self, x_batch):
        x_padded_batch = []
        y_batch = []

        for i, x in enumerate(x_batch):
            if len(x) < self.n:
                pad_len = self.n - len(x)
                padding = torch.full((pad_len,) + x.size()[1:], self.tokenizer.pad_token_id, dtype=x.dtype, device=x.device)
                x_padded = torch.cat((padding, x), dim=0)
            elif len(x) > self.n:
                x_padded = x[-self.n:]
            else:
                x_padded = x

            y = torch.cat((x_padded[1:], torch.tensor([-100])), dim=0)

            x_padded_batch.append(x_padded)
            y_batch.append(y)

        return torch.stack(x_padded_batch), torch.stack(y_batch)

    def decode(self, x, path):
        midi_file = self.tokenizer.decode(x)
        midi_file.dump_midi(Path(path))

class DataConstructor():
    def get_midi_files(root_dir):
        print("loading...")
        midi_paths = list(Path(root_dir).resolve().glob("**/*.mid"))
        return midi_paths
    
    def get_tokens_files(root_dir):
        print("loading...")
        midi_paths = list(Path(root_dir).resolve().glob("**/*.json"))
        return midi_paths
    
    def pretokenize(root_dir, tokenizer, save_dir):
        root_dir = Path(root_dir).resolve()  # Ensure root_dir is an absolute path
        save_dir = Path(save_dir).resolve()  # Ensure save_dir is an absolute path
    
        tokenizer.tokenize_dataset(root_dir,
                                  save_dir)
    
    def split_and_augment(root_dir, save_dir, partition, tokenizer, n):
        midi_files = []
        
        for p in root_dir:
            midi_files += DataConstructor.get_midi_files(p)
        
        midi_paths = filter_dataset(midi_files)
        random.shuffle(midi_files)
        
        midi_train = midi_files[:int(partition['train'] * len(midi_files))]
        midi_validation = midi_files[int(partition['train'] * len(midi_files)):int(partition['train'] * len(midi_files)) + int(partition['validation'] * len(midi_files))]
        midi_test = midi_files[-(len(midi_files)-len(midi_train)-len(midi_validation)):]
        
        print(len(midi_files))
        
        for files_paths, subset_name in (
                (midi_train, "train"), (midi_validation, "valid"), (midi_test, "test")
        ):
            for i, p in enumerate(files_paths):
                print(p)
                try:
                    chunk_dir = Path(f"{save_dir}", f"{subset_name}", f"{i}")
                    split_files_for_training(
                        files_paths=[p],
                        tokenizer=tokenizer,
                        save_dir=chunk_dir,
                        max_seq_len=n
                    )
                    
                    augment_dataset(
                        chunk_dir,
                        pitch_offsets=[-12, 12],
                        velocity_offsets=[-4, 4],
                        duration_offsets=[-0.5, 0.5],
                    )
                    
                except:
                    print(f"\n\n\n NOT TAKEN INTO ACCOUNT : {p}\n\n\n")
    
    def get_set(save_dir, n, tokenizer):

        midi_train = DataConstructor.get_midi_files(f"{save_dir}/train")
        print(f"Train dataset : {len(midi_train)} elements.",flush=True)
        midi_validation = DataConstructor.get_midi_files(f"{save_dir}/valid")
        print(f"Validation dataset : {len(midi_validation)} elements.",flush=True)
        midi_test = DataConstructor.get_midi_files(f"{save_dir}/test")
        print(f"Test dataset : {len(midi_test)} elements.",flush=True)
        
        kwargs_dataset = {"max_seq_len": n, "tokenizer": tokenizer, "bos_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer["EOS_None"]}
        dataset_train = DatasetMIDI(midi_train, **kwargs_dataset)
        dataset_valid = DatasetMIDI(midi_validation, **kwargs_dataset)
        dataset_test = DatasetMIDI(midi_test, **kwargs_dataset)

        return dataset_train, dataset_valid, dataset_test
    
    def get_JSONset(save_dir, n, tokenizer):

        midi_train = DataConstructor.get_tokens_files(f"{save_dir}/train")
        print(f"Train dataset : {len(midi_train)} elements.",flush=True)
        midi_validation = DataConstructor.get_tokens_files(f"{save_dir}/valid")
        print(f"Validation dataset : {len(midi_validation)} elements.",flush=True)
        midi_test = DataConstructor.get_tokens_files(f"{save_dir}/test")
        print(f"Test dataset : {len(midi_test)} elements.",flush=True)
        
        kwargs_dataset = {"max_seq_len": n, "bos_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer["EOS_None"]}
        dataset_train = DatasetJSON(midi_train, **kwargs_dataset)
        dataset_valid = DatasetJSON(midi_validation, **kwargs_dataset)
        dataset_test = DatasetJSON(midi_test, **kwargs_dataset)

        return dataset_train, dataset_valid, dataset_test
    
    
    def check_set(root_dir, tokenizer):
        root_dir = Path(root_dir)
        # Iterate over all files in the directory and its subdirectories
        idx = 0
        for file_path in root_dir.glob('**/*.mid'):
            idx += 1
            try:
                # Tokenize MIDI file
                tokens = tokenizer.encode(str(file_path))
                # Optionally, you can save the tokens or perform other operations here
                print(f"Successfully tokenized {file_path} [{idx}]\t\t\t\t\t\t\t\t\t", end='\r')
            except Exception as e:
                print(f"\n")
                print(f"Error processing {file_path}: {e}")
                print(f"\n")

class Mozart(nn.Module):
    def __init__(self, n = 1024, embedding_factor = 0.5, n_blocks = 5, n_heads = 5, dropout = 0.3, eval_every=100, eval_iters=10, partition = {'train':0.6, 'validation':0.2, 'test':0.2}, verbose=True, verbose_iter = 5, force_cpu=False, compose_length = 1024, batch_size=32, num_epochs=50000, ID=None, nm_workers=2):
        super().__init__()

        self.verbose = verbose
        self.verbose_iter = verbose_iter
        print("Initializing translater...",flush=True)
        self.translater = Translater(n)
        print("Translater initialized !",flush=True)
        
        self.n = n
        self.nbr_tokens = self.translater.tokenizer.vocab_size
        n_embedding = int(embedding_factor * self.nbr_tokens)
        self.n_embedding = n_embedding
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_size = n_embedding
        self.dropout = dropout
        self.eval_every = eval_every
        self.eval_iters = eval_iters
        
        print("Initializing transformer...",flush=True)
        self.transformer = Transformer(self.nbr_tokens, n, n_embedding, n_blocks, n_heads, self.head_size, dropout)
        print("Transformer initialized !",flush=True)


        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = 3e-4
        
        self.compose_length = compose_length
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if force_cpu:
            self.device = 'cpu'
        
        if ID == None:
            self.ID = random.randint(0, 100)
        else:
            self.ID = ID
        
        self.directory = "wandb_follow/"
        self.checkpoint = f"{self.directory}Mozart-{self.ID}/"
        
        # Check if the directory exists
        if not os.path.exists(self.checkpoint):
            # Create the directory if it doesn't exist
            os.makedirs(self.checkpoint)


        self.save_iter = 5000

        wandb.init(
            dir=self.directory,
            project="Mozart-GPT",
            config={
                "model": f"MozartV6-{self.ID}",
                "batch_size": self.batch_size,
                "block_size": self.n,
                "n_embedding": self.n_embedding,
                "n_blocks": self.n_blocks,
                "n_head": self.n_heads,
                "dropout": self.dropout,
                "num_epochs": self.num_epochs,
                "learning_rate": self.lr,
                "eval_every": self.eval_every,
                "eval_iters": self.eval_iters,
                "nbr_tokens":  self.nbr_tokens
            }
        )
        
        self.path_midi = f"./outputs/{self.ID}/"
        
        if not os.path.exists(self.path_midi):
            os.makedirs(self.path_midi)
        
        print("Moving transformer...",flush=True)
        self.transformer = self.transformer.to(self.device)
        print("Transformer moved !",flush=True)
        print("\n\n= \t = \t = \t =")
        print(f"Mozart {self.ID} will train on {torch.cuda.device_count()} {self.device} and has {sum(p.numel() for p in self.transformer.parameters())/1e6} M parameters", flush=True)
        print(f"Mozart use REMI as the tokenizer with {self.nbr_tokens} tokens", flush=True)
        print("= \t = \t = \t =")

        if torch.cuda.device_count() > 1:
            print(f"Mozart {self.ID} has found {torch.cuda.device_count()} GPU and will litteraly clone itself to train way faster !", flush=True)
            self.transformer = nn.DataParallel(self.transformer)
                
        
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()

        print("Getting the sets :", flush=True)
        self.train_dataset, self.validation_dataset, self.test_dataset = DataConstructor.get_JSONset('./dataset/v3_augmented/tokens', self.n, self.translater.tokenizer)
        
        print("Making the loaders :", flush=True)
        
        #collator = DataCollator(self.translater.tokenizer["PAD_None"], copy_inputs_as_labels=True, shift_labels=True)
        
        def perso_collate_fn(batch):
            # Filter out None elements
            batch = [data for data in batch if data is not None]
            batch = [data['input_ids'][0] for data in batch if len(data['input_ids']) > 0]
            # If all elements are None, return None
            if len(batch) == 0:
                return None
            # Otherwise, pass the filtered batch to the DataCollator
            return batch

        
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=nm_workers, collate_fn=perso_collate_fn)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=nm_workers, collate_fn=perso_collate_fn)
        self.validation_loader = torch_data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=nm_workers, collate_fn=perso_collate_fn)
        print("Everythings ok !", flush=True)

        print("= \t = \t = \t =\n\n")
        
    def train(self):
        for epoch in range(self.num_epochs):
            if self.verbose and epoch % self.verbose_iter == 0:
                print(f"Mozart is training itself ! Epoch {epoch+1}/{self.num_epochs}", flush=True)

            self.transformer.train()
                        
            x,y = self.translater.get_xy(next(iter(self.train_loader)))
            
            #x = self.translater.encode(x)
            x = x.to(self.device)

            #y = self.translater.encode(y)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.transformer(x)
            outputs = torch.swapaxes(outputs, 1 , 2)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            if epoch % self.eval_every == 0:
                train_loss, validation_loss, test_loss, outputs = self.eval()
                self.eval_compose(epoch)
                wandb.log({"epoch":epoch+1, "train_loss": train_loss, "validation_loss": validation_loss, "test_loss": test_loss})
                
            if epoch % self.save_iter == 0:
                self.save(epoch)
                
        epoch = self.num_epochs
        train_loss, validation_loss, test_loss, outputs = self.eval()
        self.eval_compose(epoch)
        wandb.log({"epoch":epoch+1, "train_loss": train_loss, "validation_loss": validation_loss, "test_loss": test_loss})
        self.save(epoch)
    
    def dumb(self):
        x_batch,y = self.translater.get_xy(next(iter(self.train_loader)))
        for i, x in enumerate(x_batch):
            self.translater.decode(x.unsqueeze(0), f"{self.path_midi}/{i}.mid")
                
    def eval_compose(self, epoch):
        if self.verbose:
            print("Mozart is composing !", flush=True)
        with torch.no_grad():
            start_prompt, _ = self.translater.get_xy(next(iter(self.test_loader)))
            #start_prompt = self.translater.encode(start_prompt[0]).to(self.device)
            start_prompt = start_prompt[0].to(self.device)

            prediction = self.compose(start_prompt, self.compose_length).to('cpu')
            print(f"COMPOSED", prediction[0, self.n:], flush=True)
            self.translater.decode(prediction, f'{self.path_midi}{self.ID}_{epoch}.mid')
            self.translater.decode(prediction[:, :self.n], f'{self.path_midi}{self.ID}_{epoch}_prompt_only.mid')
            self.translater.decode(prediction[:, self.n:], f'{self.path_midi}{self.ID}_{epoch}_no_prompt.mid')
              
        print(f"Mozart has released his latest hit! You can listen to it at the following address: {self.path_midi}{self.ID}_{epoch}.mid !", flush=True)

    def save(self, epoch):
        print("Mozart is saving itself !", flush=True)
        torch.save(self.transformer.state_dict(), f"{self.checkpoint}model-{epoch}.pth")
        print(f"The brain of Mozart has been put into : {self.checkpoint}model-{epoch}.pth", flush=True)
        
        
    def eval(self):
        print("Mozart is evaluated !", flush=True)
        with torch.no_grad():
            for k in {0, 1, 2}:
                if k == 0:
                    loader = self.train_loader
                elif k == 1:
                    loader = self.validation_loader
                else:
                    loader = self.test_loader

                self.transformer.eval()

                total_loss = 0

                for i in range(self.eval_iters):
                    x, y = self.translater.get_xy(next(iter(loader)))
                    
                    #x = self.translater.encode(x)
                    x = x.to(self.device)

                    #y = self.translater.encode(y)
                    y = y.to(self.device)

                    outputs = self.transformer(x)
                    outputs = torch.swapaxes(outputs, 1 , 2)
                    loss = self.criterion(outputs, y)
                    total_loss += loss.item()
                if k == 0:
                    train_loss = total_loss / self.eval_iters
                elif k == 1:
                    validation_loss = total_loss / self.eval_iters
                else:
                    test_loss = total_loss / self.eval_iters

        return train_loss, validation_loss, test_loss, outputs

    def compose(self, start_prompt, length):
        # padding or cropping the start prompt such that is has a length of n
        if start_prompt.shape[-1] > self.n:
            start_prompt = start_prompt[-self.n:]
        elif start_prompt.shape[-1] < self.n:
            start_prompt = torch.cat([torch.zeros(self.n-start_prompt.shape[-1], device=self.device), start_prompt])

        start_prompt = start_prompt.unsqueeze(0)

        for i in range(length):
            x = start_prompt[:,-self.n:].to(torch.int64).to(self.device)
            y = self.transformer(x)[:, -1, :] #only the last token prediction
            y_proba = F.softmax(y, dim = 1)
            y_pred = torch.multinomial(y_proba, 1)
            start_prompt = torch.cat([start_prompt, y_pred], dim = 1)
        return start_prompt
