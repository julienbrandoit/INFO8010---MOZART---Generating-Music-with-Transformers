from mozart_v4 import Mozart as MozartV4

if __name__ == "__main__":
    mozart = MozartV4(n_blocks=6, n_heads=4, n=1024, compose_length=1024, ID="V4_first", batch_size=32, embedding_factor=0.3)
    mozart.train()