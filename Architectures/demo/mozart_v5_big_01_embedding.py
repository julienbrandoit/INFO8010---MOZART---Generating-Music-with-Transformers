from mozart_v5 import Mozart as MozartV5

if __name__ == "__main__":
    mozart = MozartV5(n_blocks=10, n_heads=8, n=1024, compose_length=1024, ID="V5_big_01_embedding", batch_size=32, embedding_factor=0.1)
    mozart.train()