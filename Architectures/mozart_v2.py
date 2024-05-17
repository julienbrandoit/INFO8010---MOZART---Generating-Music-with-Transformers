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

"""Version 2 : outputing the state of the 128 notes at each time step, with constant default velocity, for one channel"""

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
    def __init__(self, input_size, n, n_embedding, n_blocks, n_heads, head_size, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_size, n_embedding)
        self.position_embedding = nn.Embedding(n, n_embedding)
        self.blocks = nn.ModuleList([Block(n, n_embedding, n_heads, head_size, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(n_embedding, input_size)
        self.softer = torch.sigmoid
        
    def forward(self, x):
        y = self.embedding(x)    
        y_pos = self.position_embedding(torch.arange(y.shape[1], device = y.device))
        y = y + y_pos
        for block in self.blocks:
            y = block(y)
        y = self.head(y)
        y = self.softer(y)
        return y
    
class Encoder():
    def __init__(self):
        pass

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        return torch.clamp(x.permute(0, -1, -2), min=0, max=1)
        
class Decoder():
    def __init__(self, default_velocity = 96):
        self.default_velocity = default_velocity
    
    def decode(self, x):
        x = x.permute(0, -1, -2)
        return (x * self.default_velocity)
    
    def to_timetable(self, x):
        x = self.decode(x)
        timetable = torch.zeros(x.shape[0], 16, 128, x.shape[-1], dtype=torch.int8)
        
        for b in range(x.shape[0]):
            for i in range(x.shape[-1]):
                timetable[b, 0, :, i] = x[b, :, i]
        return timetable
    
class Translater():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()    
        
class SongDataset(torch_data.Dataset):
    def __init__(self, root_dir, root_list, n, dataset_type = 'train'):
        """Initializes a dataset."""
        super().__init__()
        
        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError('Error in SongDataset: dataset_type should be train, validation or test.')
        
        self.dataset_type = dataset_type
        self.root_list = root_list
        self.root_dir = root_dir
        
        self.sequence_size = n
        
        print("Dataset : ", self.dataset_type, " - ", len(self), "elements !", flush=True)

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.root_list)

    def randomize(self):
        idx = random.randint(0, len(self) - 1)
        self.data = torch.load(self.root_dir + self.root_list[idx])
        print(f"The {self.dataset_type} dataset is shuffled ! {self.root_dir + self.root_list[idx]} has been loaded : shape is {self.data.shape}. ", flush=True)

    
    def __getitem__(self, index):
        try :
            index = random.randint(0,self.data.shape[-1] - self.sequence_size)
            if index+self.sequence_size+1 >= self.data.shape[-1]:
                print(f"ERREUR !!! {self.data.shape[-1]},  {index}, {self.sequence_size}", flush=True)
                self.randomize()
                return self[0]
            d = self.data[:,index:index+self.sequence_size].to(torch.float), self.data[:,index+1:index+self.sequence_size+1].to(torch.float)
        except:
            print(f"Error while fetching data from {self.dataset_type}, retry after shuffling !", flush=True)
            self.randomize()
            return self[0]
        
        return d
    
class Mozart(nn.Module):
    def __init__(self, n = 576, input_size=128, n_embedding = 130//2, n_blocks = 5, n_heads = 5, dropout = 0.1, eval_every=100, eval_iters=10, partition = {'train':(0, 0.6), 'validation':(0.6, 0.8), 'test':(0.8, 1)}, verbose=True, verbose_iter = 5, force_cpu=False, compose_length = 2500, dataset="dataset/data_sliced_one/", shuffle_iter=1, batch_size=32, ID = None, num_epochs=50000):
        super().__init__()

        self.verbose = verbose
        self.verbose_iter = verbose_iter
        
        self.n = n
        self.input_size = input_size
        self.n_embedding = n_embedding
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_size = n_embedding
        self.dropout = dropout
        self.eval_every = 100
        self.eval_iters = 10
        
        self.compose_length = compose_length
        
        self.transformer = Transformer(input_size, n, n_embedding, n_blocks, n_heads, self.head_size, dropout)
        self.translater = Translater()
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = 3e-4
        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if force_cpu:
            self.device = 'cpu'
        
        if ID == None:
            self.ID = random.randint(0, 100)
        else:
            self.ID = ID
            
        self.directory = "wandb_follow/"
        self.checkpoint = f"{self.directory}/Mozart-{self.ID}.pt"

        wandb.init(
            dir=self.directory,
            project="Mozart-GPT",
            config={
                "model": f"MozartV2-{self.ID}",
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
            }
        )
        
        self.path_midi = f"./outputs/{self.ID}/"
        
        if not os.path.exists(self.path_midi):
            os.makedirs(self.path_midi)
        
        self.transformer = self.transformer.to(self.device)
        print("\n\n= \t = \t = \t =")
        print(f"Mozart {self.ID} will train on {torch.cuda.device_count()} {self.device} and has {sum(p.numel() for p in self.transformer.parameters())/1e6} M parameters", flush=True)
        print("= \t = \t = \t =")

        
        if torch.cuda.device_count() > 1:
            print(f"Mozart {self.ID} has found {torch.cuda.device_count()} GPU and will litteraly clone itself to train way faster !", flush=True)
            self.transformer = nn.DataParallel(self.transformer)
                
        
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr = self.lr)
        self.criterion = lambda y_hat, y  : F.binary_cross_entropy(y_hat, y, reduction='mean')

        self.dataset = dataset
        self.shuffle_iter = shuffle_iter
        
        files = []
        for file_name in os.listdir(dataset):
            if file_name.endswith('.pt'):
                files.append(file_name)
                
        random.shuffle(files)
        
        train_end = int(len(files)*partition['train'][1])
        validation_end = int(len(files)*partition['validation'][1])
        
        self.train_dataset = SongDataset(self.dataset, files[:train_end], self.n, dataset_type='train')
        self.validation_dataset = SongDataset(self.dataset, files[train_end:validation_end], self.n, dataset_type='validation')
        self.test_dataset = SongDataset(self.dataset, files[validation_end:], self.n, dataset_type='test')

        self.train_dataset.randomize()
        self.validation_dataset.randomize()
        self.test_dataset.randomize()
        
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.validation_loader = torch_data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        print("= \t = \t = \t =\n\n")
        
    def train(self):
        for epoch in range(self.num_epochs + 1):
            if epoch % self.shuffle_iter == 0:
                self.train_dataset.randomize()
            if self.verbose and epoch % self.verbose_iter == 0:
                print(f"Mozart is training itself ! Epoch {epoch+1}/{self.num_epochs}", flush=True)

            self.transformer.train()
            
            x,y = next(iter(self.train_loader))
            
            x = self.translater.encoder.encode(x)
            x = x.to(self.device)

            y = self.translater.encoder.encode(y)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.transformer(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            if epoch % self.eval_every == 0:
                train_loss, validation_loss, test_loss, outputs = self.eval()
                self.eval_compose(epoch)
                wandb.log({"epoch":epoch+1, "train_loss": train_loss, "validation_loss": validation_loss, "test_loss": test_loss})

    def eval_compose(self, epoch):
        if self.verbose:
            print("Mozart is composing !", flush=True)
        with torch.no_grad():
            start_prompt, _ = next(iter(self.test_loader))
            start_prompt = self.translater.encoder.encode(start_prompt[0]).to(self.device)

            prediction = self.compose(start_prompt, self.compose_length)
            prediction = self.translater.decoder.to_timetable(prediction)
            
            prediction_midi = pp.tensor_to_midi(prediction[0].to('cpu').numpy())
            prediction_midi.save(f'{self.path_midi}{self.ID}_{epoch}.mid')
        print(f"Mozart has released his latest hit! You can listen to it at the following address: {self.path_midi}{self.ID}_{epoch}.mid !", flush=True)

    def eval(self):
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
                    
                    if i % self.shuffle_iter == 0:
                        if k == 0:
                            self.train_dataset.randomize()
                        elif k == 1:
                            self.validation_dataset.randomize()
                        else:
                            self.test_dataset.randomize()

                    x, y = next(iter(loader))
                    
                    x = self.translater.encoder.encode(x)
                    x = x.to(self.device)

                    y = self.translater.encoder.encode(y)
                    y = y.to(self.device)

                    outputs = self.transformer(x)
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
        if start_prompt.shape[1] > self.n:
            start_prompt = start_prompt[:, -self.n:, :]
        elif start_prompt.shape[1] < self.n:
            print("resizing", flush=True)
            start_prompt = torch.cat([torch.zeros((1, self.n-start_prompt.shape[1], 128), device=self.device), start_prompt],  dim=1)

        for i in range(length):
            x = start_prompt[:, -self.n:,:].to(torch.float).to(self.device)
            y = self.transformer(x)[:, -1, :] #only the last token prediction
            y_pred = torch.bernoulli(y).unsqueeze(1)
            start_prompt = torch.cat([start_prompt, y_pred], dim = 1)
        return start_prompt