import glob
import torch
import random
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from mask_cyclegan_vc.maskcyclegan import MaskCycleGAN

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wav_path_a', type=str)
parser.add_argument('--wav_path_b', type=str)
parser.add_argument('--wandb_name', type=str)

args = parser.parse_args()

SAMPLE_RATE = 16000

class UnpairDataset(Dataset):
    def __init__(self, wav_path_a, wav_path_b, is_train=True):
        super().__init__()
        list_a = glob.glob(wav_path_a + '/*.wav')
        list_b = glob.glob(wav_path_b + '/*.wav')

        self.max_len = max(len(self.wav_list_a), len(self.wav_list_b))

        n_fft = 510
        self.stft_conf = dict(
            n_fft=n_fft, hop_length=160,
            win_length=400,
            window=torch.hann_window(window_length=400), normalized=False,
            onesided=True
        )

        self.is_train = is_train

        self.cut_dim = 128
        self.img_dim = (n_fft // 2 + 1, self.cut_dim)
        self.mask_width = [0, 80]

    def __len__(self):
        return self.max_len
    
    def read_data(self, path):
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        spec = torch.stft(wav, **self.stft_conf, return_complex=True)

        return torch.stack([spec.real, spec.imag], dim=1)[0]

    def random_cut(self, spec):
        end = spec.size(-1) - self.cut_dim
        idx = np.random.randint(0, end + 1)
        return spec[:, :, idx:idx + self.cut_dim]

    def gen_mask(self, mask):
        mask_width = random.randint(self.mask_width[0], self.mask_width[1])
        start_idx = random.randint(0, self.cut_dim - mask_width)
        end_idx = start_idx + mask_width
        mask[:, :, start_idx:end_idx] = 0
        return mask

    def __getitem__(self, idx):
        idx_a = idx % len(self.wav_list_a)
        idx_b = idx % len(self.wav_list_b)

        wav_path_a = self.wav_list_a[idx_a]
        wav_path_b = self.wav_list_b[idx_b]

        spec_a = self.read_data(wav_path_a)
        spec_b = self.read_data(wav_path_b)

        spec_a = self.random_cut(spec_a)
        spec_b = self.random_cut(spec_b)

        mask_a = torch.ones(1, *self.img_dim)
        mask_b = torch.ones(1, *self.img_dim)

        if self.is_train:
            mask_a = self.gen_mask(mask_a)
            mask_b = self.gen_mask(mask_b)

        return spec_a, mask_a, spec_b, mask_b

# wav_path_a = '/Users/tuanio/Downloads/l2arctic_release_v5.0/ABA/wav'
# wav_path_b = '/Users/tuanio/Downloads/l2arctic_release_v5.0/PNV/wav'

wav_path_a = args.wav_path_a
wav_path_b = args.wav_path_b

dataset = UnpairDataset(wav_path_a, wav_path_b, is_train=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

mask_cyclegan_vc = MaskCycleGAN()

wandb = WandbLogger(project='5project', name=args.wandb_name)

trainer = pl.Trainer(
    max_steps=500_000,
    logger=wandb
)
trainer.fit(model=mask_cyclegan_vc, train_dataloaders=dataloader)