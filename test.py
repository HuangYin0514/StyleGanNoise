import torch
import numpy as np
from run import *

if __name__ == "__main__":
    data = '../../gan/custom_dataset'
    results_dir = './GoodResult/results'
    models_dir = './GoodResult/models'
    name = 'mytest'
    new = False
    load_from = -1
    image_size = 64
    network_capacity = 16
    transparent = False
    batch_size = 3
    gradient_accumulate_every = 5
    num_train_steps = 100000
    learning_rate = 2e-4
    num_workers = None
    save_every = 10000
    generate = False
    num_image_tiles = 8
    trunc_psi = 0.6

    train = Trainer('1111',
                    results_dir,
                    models_dir,
                    batch_size=batch_size,
                    gradient_accumulate_every=gradient_accumulate_every,
                    image_size=image_size,
                    network_capacity=network_capacity,
                    transparent=transparent,
                    lr=learning_rate,
                    num_workers=num_workers,
                    save_every=save_every,
                    trunc_psi=trunc_psi)
    model = train.init_GAN()
    train.GAN.S
