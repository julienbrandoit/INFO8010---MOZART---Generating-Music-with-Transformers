from mozart_v1 import Mozart as MozartV1

if __name__ == "__main__":
    mozart = MozartV1(n = 288*2, n_blocks=2, batch_size=128, n_heads=2, dropout=0.3, compose_length=2500, num_epochs=100000, ID="small_1_2")
    mozart.train()