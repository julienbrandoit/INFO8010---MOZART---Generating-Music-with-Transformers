from mozart_v6 import Mozart as MozartV6

if __name__ == "__main__":
    mozart = MozartV6(n_blocks=12, n_heads=8, n=1024, embedding_factor=0.35, compose_length=1024, ID="V6_huge", batch_size=256)
    mozart.train()