from mozart_v2 import Mozart as MozartV2

if __name__ == "__main__":
    mozart = MozartV2(n = int(288), n_blocks=10, batch_size=96, n_heads=8, dropout=0.3, compose_length=2500, num_epochs=100000, ID = "medium_2_cross_entropy", shuffle_iter = 20)
    mozart.train()