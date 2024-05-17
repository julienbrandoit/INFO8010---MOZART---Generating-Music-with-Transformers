from mozart_v5 import Mozart as MozartV5

if __name__ == "__main__":
    mozart = MozartV5(n_blocks=6, n_heads=4, n=1024, compose_length=1024, ID="V5_first", batch_size=32, embedding_factor=0.3)
    mozart.train()