import json
from math import floor, log2
from random import random
from shutil import rmtree
import multiprocessing
import torch.backends.cudnn as cudnn

import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F

import torchvision

from pathlib import Path
from utils import *
from datasets import Dataset
from net import StyleGAN2

num_cores = multiprocessing.cpu_count()

# constants
EPS = 1e-8
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# speed up
cudnn.benchmark = True


class Trainer():
    def __init__(self,
                 name,
                 results_dir,
                 models_dir,
                 image_size,
                 network_capacity,
                 transparent=False,
                 batch_size=4,
                 mixed_prob=0.9,
                 gradient_accumulate_every=1,
                 lr=2e-4,
                 num_workers=None,
                 save_every=1000,
                 trunc_psi=0.6,
                 *args,
                 **kwargs):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(
        ), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = 0

        self.gradient_accumulate_every = gradient_accumulate_every

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr=self.lr,
                             image_size=self.image_size,
                             network_capacity=self.network_capacity,
                             transparent=self.transparent,
                             *args,
                             **kwargs)
        self.GAN.to(device)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config(
        ) if not self.config_path.exists() else json.loads(
            self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']

        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'network_capacity': self.network_capacity,
            'transparent': self.transparent
        }

    def set_data_src(self, folder):
        self.dataset = Dataset(folder,
                               self.image_size,
                               transparent=self.transparent)
        self.loader = cycle(
            data.DataLoader(self.dataset,
                            num_workers=default(self.num_workers, num_cores),
                            batch_size=self.batch_size,
                            drop_last=True,
                            shuffle=True,
                            pin_memory=True))

    def train(self):
        assert self.loader is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_noise_loss = torch.tensor(0.).to(device)

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        # trian noise layers
        get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
        style = get_latents_fn(batch_size, num_layers, latent_dim)
        noise = custom_image_nosie(batch_size, 100)

        w_space = latent_to_w(self.GAN.S, style)
        w_styles = styles_def_to_tensor(w_space)
        noise_space = latent_to_nosie(self.GAN.N, noise)

        generated_images = self.GAN.G(w_styles, noise_space)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        # if any(torch.isnan(l) for l in (total_noise_loss)):
        #     print(
        #         f'NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}'
        #     )
        #     self.load(checkpoint_num)
        #     raise NanException

        # periodically save results

        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.steps % 1000 == 0 or (self.steps % 100 == 0
                                      and self.steps < 2500):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=8, trunc=1.0):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        def generate_images(stylizer, generator, latents, noise):
            w = latent_to_w(stylizer, latents)
            w_styles = styles_def_to_tensor(w)
            generated_images = evaluate_in_chunks(self.batch_size, generator,
                                                  w_styles, noise)
            generated_images.clamp_(0., 1.)
            return generated_images

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        latents = noise_list(num_rows**2, num_layers, latent_dim)
        n = image_noise(num_rows**2, image_size)

        # regular
        generated_images = generate_images(self.GAN.S, self.GAN.G, latents, n)
        torchvision.utils.save_image(generated_images,
                                     str(self.results_dir / self.name /
                                         f'{str(num)}.{ext}'),
                                     nrow=num_rows)

        # moving averages
        generated_images = self.generate_truncated(self.GAN.SE,
                                                   self.GAN.GE,
                                                   latents,
                                                   n,
                                                   trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images,
                                     str(self.results_dir / self.name /
                                         f'{str(num)}-ema.{ext}'),
                                     nrow=num_rows)

        # mixing regularities
        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(
                np.concatenate([
                    init_dim * np.arange(n_tile) + i for i in range(init_dim)
                ])).to(device)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE,
                                                   self.GAN.GE,
                                                   mixed_latents,
                                                   n,
                                                   trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images,
                                     str(self.results_dir / self.name /
                                         f'{str(num)}-mr.{ext}'),
                                     nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi=0.6, num_image_tiles=8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).to(device)
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(
            self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    def print_log(self):
        print(
            f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {self.pl_mean:.2f}'
        )

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [
                p for p in Path(self.models_dir / self.name).glob('model_*.pt')
            ]
            saved_nums = sorted(
                map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        load_model_name = f'model_{name}.pt'
        self.GAN.load_state_dict(
            torch.load(load_model_name,
                       map_location=torch.device(device)))
        print(load_model_name)
