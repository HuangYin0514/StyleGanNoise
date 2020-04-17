#!/usr/bin/env python
import fire
from retry.api import retry_call
from tqdm import tqdm
from stylegan2_pytorch import Trainer, NanException
from datetime import datetime


def train_from_folder(data='../../gan/custom_dataset',
                      results_dir='./GoodResult/results',
                      models_dir='./GoodResult/models',
                      name='mytest',
                      new=False,
                      load_from=-1,
                      image_size=64,
                      network_capacity=16,
                      transparent=False,
                      batch_size=3,
                      gradient_accumulate_every=5,
                      num_train_steps=100000,
                      learning_rate=2e-4,
                      num_workers=None,
                      save_every=10000,
                      generate=False,
                      num_image_tiles=8,
                      trunc_psi=0.6):
    model = Trainer(name,
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

    if not new:
        model.load(load_from)
    else:
        model.clear()

    if generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.evaluate(samples_name, num_image_tiles)
        print(
            f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    model.set_data_src(data)

    train_now = datetime.now().timestamp()
    for _ in tqdm(range(num_train_steps - model.steps), mininterval=10., desc=f'{name}<{data}>'):
        #train
        retry_call(model.train, tries=3, exceptions=NanException)
        # stop time
        if _ % 500 == 0:
            if datetime.now().timestamp() - train_now > 29880:
                break
        if _ % 50 == 0:
            model.print_log()


if __name__ == "__main__":
    fire.Fire(train_from_folder)
