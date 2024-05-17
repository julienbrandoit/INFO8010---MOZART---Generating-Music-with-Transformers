from mozart_v6 import Mozart as MozartV6

if __name__ == "__main__":
    mozart = MozartV6(n_blocks=8, n_heads=5, n=1024, compose_length=1024, ID="V6_big_third_on_v3", batch_size=64, num_epochs=100000)
    mozart.train()