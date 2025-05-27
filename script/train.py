import sys
from model.model import AW
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


import lightning.pytorch as pl
import torch.optim as optim
import math

from torch.utils.data import Dataset, DataLoader, RandomSampler

from dataloader.dataset_combine import WMDataset, collate_fn



#you need to define your dataset and dataloader first

pl.seed_everything(888)
model=AW(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8)



checkpoint_callback = ModelCheckpoint(
        dirpath='path_to_your_ckpt',
        filename='mymodel-{step:02d}-{loss:.2f}-{val_loss:.2f}',
        save_top_k=100,
        every_n_train_steps=2000,
        monitor='loss',
        mode='min',
        save_last=True
    )
trainer=pl.Trainer(
    profiler="simple",
    logger=TensorBoardLogger(name='my_model',save_dir='path_to_your_log'),
    accelerator='gpu',
    num_nodes=1,
    devices=8,
    log_every_n_steps=50,
    precision="32",
    max_steps=1000000,
    callbacks=[checkpoint_callback],
    strategy="ddp_find_unused_parameters_true"
    )
trainer.fit(model, dataloader_train, dataloader_val)