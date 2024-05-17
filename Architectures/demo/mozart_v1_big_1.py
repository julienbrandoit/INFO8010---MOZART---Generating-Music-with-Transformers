from mozart_v1 import Mozart as MozartV1

if __name__ == "__main__":
    mozart = MozartV1(n = 288*2, n_blocks=10, batch_size=64, n_heads=7, dropout=0.3, compose_length=2500, num_epochs=100000, ID = "big_1_2")
    mozart.train()