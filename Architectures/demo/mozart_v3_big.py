from mozart_v3 import Mozart as MozartV3

if __name__ == "__main__":
    mozart = MozartV3(n_blocks=8, n_heads=5, n=1024, compose_length=1024, ID="V3_big", batch_size=64)
    mozart.train()