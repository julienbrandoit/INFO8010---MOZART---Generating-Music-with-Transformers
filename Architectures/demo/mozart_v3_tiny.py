from mozart_v3 import Mozart as MozartV3

if __name__ == "__main__":
    mozart = MozartV3(n_blocks=6, n_heads=4, n=1024, compose_length=1024, ID="V3_first", batch_size=32)
    mozart.train()