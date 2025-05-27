import sys
sys.path.append('../')
from model.model import AW

import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pesq import pesq
import lightning.pytorch as pl
import torch.optim as optim
import math
import torchaudio
import pickle
import tqdm
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast
import json
import sys

import torch


def signal_noise_ratio(original, signal_watermarked):
    original,signal_watermarked = original.cpu().numpy(), signal_watermarked.cpu().numpy()
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:  #
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength
    ratio = max(1e-10, ratio)
    return 10 * np.log10(ratio)

def ber_loss(payload_restored, payload):
    payload_restored_binary = (payload_restored > 0.5).float()
    payload_binary = (payload > 0.5).float().squeeze(0)
    error_bits = torch.sum(torch.abs(payload_restored_binary - payload_binary))
    ber = error_bits / payload.numel()
    return ber


model = AW(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8)
ckpt = torch.load("ckpt/model.ckpt", map_location=torch.device('cpu'))["state_dict"]
model.load_state_dict(ckpt, strict=False)
model.eval()


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


model.to(device)


wavpath='test.wav' # or change to your own path
signal, sr = torchaudio.load(wavpath)
if sr != 16000:
    signal = torchaudio.transforms.Resample(sr, 16000)(signal)
signal = signal.to(device)
wm0 = torch.randint(0,2,(1,32)).to(device).float()
key0=torch.randint(1, 10, (8,)).to(device).float()
wm1 = torch.randint(0,2,(1,32)).to(device).float()
# make sure wm0 != wm1
while torch.equal(wm0, wm1):
    wm1 = torch.randint(0,2,(1,32)).to(device).float()
key1=torch.randint(1, 10, (8,)).to(device).float()
# make sure key0 != key1
while torch.equal(key0, key1):
    key1 = torch.randint(1, 10, (8,)).to(device).float()

# to test the watermark
signal_input = signal[:,0:16000]
with torch.no_grad():
    watermark_once, _ = model.generator.encode(signal_input, wm0, key0)
    watermark_restored, _ = model.generator.decode(watermark_once, key0)
ber1_0 = ber_loss(watermark_restored, wm0)
#you can decode with another key and check the ber
with torch.no_grad():
    watermark_restored1, _ = model.generator.decode(watermark_once, key1)
ber1_1 = ber_loss(watermark_restored1, wm0)
pesq1_1 = pesq(16000, signal_input[0].cpu().numpy(), watermark_once[0].cpu().numpy(), 'wb')
snr1_1 = signal_noise_ratio(signal_input[0], watermark_once[0])
print("once watermark and decode with key0 and compare with watermark 0 BER: ", ber1_0.item())
print("once watermark and decode with key1 and compare with watermark 0 BER: ", ber1_1.item())
print("once watermark pesq: ", pesq1_1)
print("once watermark snr: ", snr1_1)

print("once watermark test end")

# to test the twice watermark
signal_input = signal[:,0:16000]
with torch.no_grad():
    watermark_once, _ = model.generator.encode(signal_input, wm0, key0)
    watermark_twice, _ = model.generator.encode(watermark_once, wm1, key1)

    watermark_restored0, _ = model.generator.decode(watermark_twice, key0)
    watermark_restored1, _ = model.generator.decode(watermark_twice, key1)

ber2_1 = ber_loss(watermark_restored0, wm0)
ber2_2 = ber_loss(watermark_restored1, wm1)
pesq2_1 = pesq(16000, signal_input[0].cpu().numpy(), watermark_once[0].cpu().numpy(), 'wb')
snr2_1 = signal_noise_ratio(signal_input[0], watermark_once[0])
pesq2_2 = pesq(16000, signal_input[0].cpu().numpy(), watermark_twice[0].cpu().numpy(), 'wb')
snr2_2 = signal_noise_ratio(signal_input[0], watermark_twice[0])
print("twice watermark and decode with key0 and compare with watermark 0 BER: ", ber2_1.item())
print("twice watermark and decode with key1 and compare with watermark 1 BER: ", ber2_2.item())
print("once watermark pesq: ", pesq2_1)
print("once watermark snr: ", snr2_1)
print("twice watermark pesq: ", pesq2_2)
print("twice watermark snr: ", snr2_2)
#you can also decode with another key and check the ber
key2=torch.randint(1, 10, (8,)).to(device).float()
if torch.equal(key0, key2) or torch.equal(key1, key2):
    key2=torch.randint(1, 10, (8,)).to(device).float()
with torch.no_grad():
    watermark_restored2, _ = model.generator.decode(watermark_twice, key2)

ber_wrongkey_0 = ber_loss(watermark_restored2, wm0)
ber_wrongkey_1 = ber_loss(watermark_restored2, wm1)
print("twice watermark and decode with wrong key and compare with watermark 0 BER: ", ber_wrongkey_0.item())
print("twice watermark and decode with wrong key and compare with watermark 1 BER: ", ber_wrongkey_1.item())

#optional you can save the wav and listen to the difference,the save file will be embed_single_watermark.wav and embed_double_watermark.wav
# savepath="./"
# model.add_watermark_save(signal, wm0, key0, wm1, key1, savepath)


