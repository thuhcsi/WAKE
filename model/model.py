import torch
from torch import Tensor
from lightning.pytorch import LightningModule
import sys
from utils.audio_to_mel import Audio2Mel
from attack.attack import AudioEffects
from module.hinet_key import Hinet
from module.detector import base_discriminator as Detector
import random
import torch.nn as nn
import soundfile
import os
from pesq import pesq

from module.predict_module import PredictiveModule


class Generator(nn.Module):
    def __init__(self, num_point, num_bit, n_fft, hop_length, num_layers,device):
        super(Generator, self).__init__()
        self.device=device
        self.attack_layer = AudioEffects
        self.hinet = Hinet(num_layers=num_layers).to(device)
        self.watermark_fc = torch.nn.Linear(num_bit, num_point).to(device)
        self.watermark_fc_back = torch.nn.Linear(num_point, num_bit).to(device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.addmodule=PredictiveModule(41,41)

    def forward(self, x, payload,key,payload1,key1,choice,i=1):
        x = x.to(self.device)
        watermark_signal, msg_remain = self.encode(x, payload,key)

        i = random.randint(1, 14)
        wav_add_wm = watermark_signal
        
        if choice==0:
            wav_add_wm_attack = wav_add_wm.unsqueeze(1)
            wav_add_wm_attack = self.attack_layer.choice(wav_add_wm_attack, attack_id=i)
            wav_add_wm_attack = wav_add_wm_attack.squeeze(1)
            payload_restored,wav_recover=self.decode(wav_add_wm_attack,key)

            return wav_add_wm, wav_add_wm_attack, payload_restored

        elif choice==1:
            wav_add_wm1, msg_remain = self.encode(wav_add_wm, payload1,key1)
            wav_add_wm_attack = wav_add_wm1.unsqueeze(1)
            wav_add_wm_attack = self.attack_layer.choice(wav_add_wm_attack, attack_id=i)
            wav_add_wm_attack = wav_add_wm_attack.squeeze(1)
            payload_restored,wav_recover=self.decode(wav_add_wm_attack,key)
            payload_restored1,wav_recover1=self.decode(wav_add_wm_attack,key1)

            return wav_add_wm,wav_add_wm1, wav_add_wm_attack, payload_restored,payload_restored1
    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
        return tmp
    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)
    def encode(self, signal, message,key):
        signal_fft = self.stft(signal)
        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)
        signal_wmd_fft, msg_remain = self.enc_dec(signal_fft, message_fft,key, rev=False)
        signal_wmd = self.istft(signal_wmd_fft)
        msg_remain = self.istft(msg_remain)
        return signal_wmd, msg_remain

    def decode(self, signal,key):
        signal_fft = self.stft(signal)
        payload=signal_fft.permute(0,2,1,3)
        payload = self.addmodule(payload)
        payload = payload.permute(0,2,1,3)
        wav_recover, message_restored_fft = self.enc_dec(signal_fft, payload,key, rev=True)

        message_restored_expanded = self.istft(message_restored_fft)
        wav_recover = self.istft(wav_recover)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        return message_restored_float, wav_recover

    def enc_dec(self, signal, watermark,key, rev):
        signal = signal.permute(0, 3, 2, 1)
        watermark = watermark.permute(0, 3, 2, 1)
        signal2, watermark2 = self.hinet(signal, watermark,key, rev)
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.detector = Detector(2, 32, 1)

    def forward(self, wav, wav_add_wm):
        wav = wav.unsqueeze(1)
        wav_add_wm = wav_add_wm.unsqueeze(1)
        wav_judge=self.detector(wav)

        watermark_judge = self.detector(wav_add_wm)

        return wav_judge,watermark_judge



class AW(LightningModule):
    def __init__(self, num_point, num_bit, n_fft, hop_length, num_layers):
        super().__init__()
        self.generator = Generator(num_point, num_bit, n_fft, hop_length, num_layers,self.device)
        self.discriminator = Discriminator()
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
        self.payload_loss=torch.nn.MSELoss()
        self.automatic_optimization = False
    
    def wav_loss(self,input_wav, output_wav):

        l1_loss = torch.nn.MSELoss()(input_wav, output_wav)
        l_f = torch.tensor([0.0]).to(input_wav.device)
        for i in range(5, 12): #e=5,...,11
            fft = Audio2Mel(n_fft=2 ** i,win_length=2 ** i, hop_length=(2 ** i) // 4, n_mel_channels=64, sampling_rate=16000)
            fft_input = fft(input_wav)
            fft_output = fft(output_wav)
            weight = torch.ones(fft_input.shape).to(input_wav.device)
            len_w = weight.shape[2]
            weight[:,:,:int(len_w * 0.03)+1] = 10
            weight[:,:,int(len_w * 0.97)-1:] = 10
            fft_input = fft_input * weight
            fft_output = fft_output * weight
            l_f = l_f + torch.nn.L1Loss()(fft_input, fft_output) + torch.nn.MSELoss()(fft_input, fft_output)


        total_loss = l1_loss+ 5* l_f

        return total_loss



    def loss_generator(self, wav, wav_add_wm, payload, payload_restored,watermark_judge_g):
        loss_wav=self.wav_loss(wav_add_wm,wav)
        self.log('loss_wav_add_wm_2_wav_origin', loss_wav, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss_payload=self.payload_loss(payload_restored,payload)
        self.log('loss_payload_restored_2_payload', loss_payload, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        g_target_label_encoded=torch.zeros(watermark_judge_g.shape[0],1).to(self.device).float()

        loss_generator_g=self.bce_with_logits_loss(watermark_judge_g,g_target_label_encoded)
        self.log('loss_generator_dis', loss_generator_g, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        loss=10*loss_wav+10*loss_payload+10*loss_generator_g
        return loss
    
    def loss_discriminator(self, wav_judge,watermark_judge):
        d_target_label_cover=torch.ones(wav_judge.shape[0],1).to(self.device).float()
        d_target_label_encoded=torch.zeros(watermark_judge.shape[0],1).to(self.device).float()
        loss_wav_judge=self.bce_with_logits_loss(wav_judge,d_target_label_cover)
        self.log('loss_origin_judge', loss_wav_judge, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss_watermark_judge=self.bce_with_logits_loss(watermark_judge,d_target_label_encoded)
        self.log('loss_watermark_judge', loss_watermark_judge, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss=loss_wav_judge+loss_watermark_judge


        return loss
    
    def forward(self, x):
        payload=torch.randint(0, 2, (x.shape[0], 32)).to(self.device).float().to(self.device)
        watermark_signal, wav_add_wm, wav_recover,payload_restored=self.generator(x, payload)
        wav_judge,watermark_judge=self.discriminator(x, wav_add_wm)
        loss_generator=self.loss_generator(x, wav_add_wm, payload, payload_restored,watermark_judge)
        loss_discriminator=self.loss_discriminator(wav_judge,watermark_judge)
        print("loss_generator:",loss_generator)
        print("loss_discriminator:",loss_discriminator)
    

    
    def test_snr(self,wav1,wav2):
        snr=10*torch.log10(torch.sum(torch.pow(wav1,2))/torch.sum(torch.pow(wav1-wav2,2)))
        return snr


    def test_pesq_snr_return_wav(self,signal,payload0,key0,payload1,key1,attack_id):
        if self.generator.device != self.device:
            self.generator.to(self.device)
            self.generator.device = self.device
            self.discriminator.to(self.device)
            print("generator to device")

        signal=signal.to(self.device)
        watermarked_1,msg_remain_1=self.generator.encode(signal, payload0,key0)
        watermarked_1_attack = watermarked_1.unsqueeze(1)
        watermarked_1_add_attack=self.generator.attack_layer.choice(watermarked_1_attack, attack_id=attack_id)
        watermarked_1_add_attack = watermarked_1_add_attack.squeeze(1)
        payload_restored_1,wav_recover_1=self.generator.decode(watermarked_1_add_attack,key0)
        try:
            pesq_score_wm=pesq(16000,signal[0].cpu().detach().numpy(), watermarked_1[0].cpu().detach().numpy(), 'wb')
        except:
            pesq_score_wm=0
        snr_wm=self.test_snr(signal[0],watermarked_1[0])

        payload_restored_1=[1 if i>0.5 else 0 for i in payload_restored_1[0].cpu().detach().numpy()]
        payload_restored_1=torch.tensor(payload_restored_1).to(self.device).float()
        payload_ber_1 = self.ber_loss(payload_restored_1, payload0)

        watermarked_2,msg_remain_2=self.generator.encode(watermarked_1, payload1,key1)
        watermarked_2_attack = watermarked_2.unsqueeze(1)
        watermarked_2_add_attack=self.generator.attack_layer.choice(watermarked_2_attack, attack_id=attack_id)
        watermarked_2_add_attack = watermarked_2_add_attack.squeeze(1)
        payload_restored_2,wav_recover_2=self.generator.decode(watermarked_2_add_attack,key1)

        
        try:
            pesq_score_wm1=pesq(16000,signal[0].cpu().detach().numpy(), watermarked_2[0].cpu().detach().numpy(), 'wb')
        except:
            pesq_score_wm1=0
        snr_wm1=self.test_snr(signal[0],watermarked_2[0])

        payload_restored_2=[1 if i>0.5 else 0 for i in payload_restored_2[0].cpu().detach().numpy()]
        payload_restored_2=torch.tensor(payload_restored_2).to(self.device).float()
        payload_ber_2 = self.ber_loss(payload_restored_2, payload1)

        payload_restored_3,wav_recover_3 = self.generator.decode(watermarked_2_add_attack,key0)

        payload_restored_3=[1 if i>0.5 else 0 for i in payload_restored_3[0].cpu().detach().numpy()]
        payload_restored_3=torch.tensor(payload_restored_3).to(self.device).float()
        payload_ber_3 = self.ber_loss(payload_restored_3, payload0)

        key3=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)
        while torch.sum(key3==key0)==8 or torch.sum(key3==key1)==8:
            key3=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)
        payload_restored_4,wav_recover_4 = self.generator.decode(watermarked_2_add_attack,key3)

        payload_restored_4=[1 if i>0.5 else 0 for i in payload_restored_4[0].cpu().detach().numpy()]
        payload_restored_4=torch.tensor(payload_restored_4).to(self.device).float()
        payload_ber_4 = self.ber_loss(payload_restored_4, payload0)
        payload_ber_5 = self.ber_loss(payload_restored_4, payload1)

        return pesq_score_wm,snr_wm,pesq_score_wm1,snr_wm1,payload_ber_1,payload_ber_2,payload_ber_3,payload_ber_4,payload_ber_5,watermarked_1,watermarked_2
    
    def add_watermark_save(self,signal,payload0,key0,payload1,key1,wavpath,attack_id=2):
        chunk_size=16000
        if self.generator.device != self.device:
            self.generator.to(self.device)
            self.generator.device = self.device
            self.discriminator.to(self.device)
            print("generator to device")
        loop = signal.shape[1] // chunk_size
        wavs_addwm_1=[]
        wavs_addwm_2=[]
        signals=[]
        with torch.no_grad():
            for i in range(loop):
                signal_chunk = signal[:, i * chunk_size:(i + 1) * chunk_size]
                payload0_chunk = payload0
                key0_chunk = key0
                payload1_chunk = payload1
                key1_chunk = key1
                pesq_score_wm,snr_wm,pesq_score_wm1,snr_wm1,payload_ber_1,payload_ber_2,payload_ber_3,payload_ber_4,payload_ber_5,watermarked_1,watermarked_2=self.test_pesq_snr_return_wav(signal_chunk,payload0_chunk,key0_chunk,payload1_chunk,key1_chunk,attack_id)
                wavs_addwm_1.append(watermarked_1)
                wavs_addwm_2.append(watermarked_2)
                signals.append(signal_chunk)
        
        wavs_addwm_1=torch.cat(wavs_addwm_1,dim=1)
        wavs_addwm_2=torch.cat(wavs_addwm_2,dim=1)
        signals=torch.cat(signals,dim=1)
        soundfile.write(wavpath+"/embed_single_watermark.wav", wavs_addwm_1[0].cpu().detach().numpy(), 16000)
        soundfile.write(wavpath+"/embed_double_watermark.wav", wavs_addwm_2[0].cpu().detach().numpy(), 16000)

    def ber_loss(self,payload_restored, payload):
        payload_restored_binary = (payload_restored > 0.5).float()
        payload_binary = (payload > 0.5).float().squeeze(0)
        error_bits = torch.sum(torch.abs(payload_restored_binary - payload_binary))
        ber = error_bits / payload.numel()
        return ber

    
    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=1e-5)
        d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=1e-5)
        return g_optimizer, d_optimizer
    
    def training_step(self, batch, batch_idx):
        g_optimizer, d_optimizer = self.optimizers()

        x=batch
        payload=torch.randint(0, 2, (x.shape[0], 32)).to(self.device).float().to(self.device)
        key=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)

        payload1=torch.randint(0, 2, (x.shape[0], 32)).to(self.device).float().to(self.device)
        key1=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)
        while torch.sum(key==key1)==8:
            key1=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)

        choice=batch_idx%2


        if choice==1:
            wav_add_wm0, wav_add_wm1,wav_add_wm_attack,payload_detect0,payload_detect1=self.generator(x, payload,key,payload1,key1,choice)
        else:
            wav_add_wm, wav_add_wm_attack,payload_detect=self.generator(x, payload,key,payload1,key1,choice)




        if choice==0:

            wav_add_wm_d = wav_add_wm.detach()

            wav_judge,watermark_judge=self.discriminator(x, wav_add_wm_d)
            loss_discriminator=self.loss_discriminator(wav_judge,watermark_judge)
            d_optimizer.zero_grad()
            self.manual_backward(loss_discriminator*2)
            d_optimizer.step()

            ############################
            # Optimize Generator
            ###########################
            watermark_judge_g=self.discriminator.detector(wav_add_wm.unsqueeze(1))
            loss_generator=self.loss_generator(x, wav_add_wm, payload, payload_detect,watermark_judge_g)
            g_optimizer.zero_grad()
            self.manual_backward(loss_generator*2)
            g_optimizer.step()

            self.log('loss_generator', loss_generator*2, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('loss_discriminator', loss_discriminator*2, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('loss', loss_generator*2+loss_discriminator*2, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        else:
            #wav_add_wm0, wav_add_wm1,wav_add_wm_attack,payload_detect0,payload_detect1=self.generator(x, payload,key,payload1,key1,choice)
            wav_add_wm0_d = wav_add_wm0.detach()
            wav_add_wm1_d = wav_add_wm1.detach()
            wav_judge0,watermark_judge0=self.discriminator(x, wav_add_wm0_d)
            wav_judge1,watermark_judge1=self.discriminator(x, wav_add_wm1_d)
            loss_discriminator0=self.loss_discriminator(wav_judge0,watermark_judge0)
            loss_discriminator1=self.loss_discriminator(wav_judge1,watermark_judge1)
            loss_discriminator=loss_discriminator0+loss_discriminator1
            d_optimizer.zero_grad()
            self.manual_backward(loss_discriminator)
            d_optimizer.step()

            ############################
            # Optimize Generator
            ###########################
            watermark_judge_g0=self.discriminator.detector(wav_add_wm0.unsqueeze(1))
            watermark_judge_g1=self.discriminator.detector(wav_add_wm1.unsqueeze(1))
            loss_generator0=self.loss_generator(x, wav_add_wm0, payload, payload_detect0,watermark_judge_g0)
            loss_generator1=self.loss_generator(x, wav_add_wm1, payload1, payload_detect1,watermark_judge_g1)
            loss_generator=loss_generator0+loss_generator1
            g_optimizer.zero_grad()
            self.manual_backward(loss_generator)
            g_optimizer.step()

            self.log('loss_generator', loss_generator, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('loss_discriminator', loss_discriminator, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('loss', loss_generator+loss_discriminator, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            



    def validation_step(self, batch, batch_idx):
        if self.generator.device != self.device:
            self.generator.to(self.device)
            self.generator.device = self.device
            self.discriminator.to(self.device)
            print("generator to device")
        x=batch
        payload=torch.randint(0, 2, (x.shape[0], 32)).to(self.device).float().to(self.device)
        key=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)

        payload1=torch.randint(0, 2, (x.shape[0], 32)).to(self.device).float().to(self.device)
        key1=torch.randint(1, 10, (8,)).to(self.device).float().to(self.device)

        choice=batch_idx%2

        if choice==1:
            wav_add_wm0, wav_add_wm1,wav_add_wm_attack,payload_detect0,payload_detect1=self.generator(x, payload,key,payload1,key1,choice)
        else:
            wav_add_wm, wav_add_wm_attack,payload_detect=self.generator(x, payload,key,payload1,key1,choice)


        if choice==0:
            wav_judge,watermark_judge=self.discriminator(x, wav_add_wm)
            loss_discriminator=self.loss_discriminator(wav_judge,watermark_judge)
            loss_generator=self.loss_generator(x, wav_add_wm, payload, payload_detect,watermark_judge)
            loss_discriminator=loss_discriminator*2
            loss_generator=loss_generator*2
        else:
            wav_judge0,watermark_judge0=self.discriminator(x, wav_add_wm0)
            wav_judge1,watermark_judge1=self.discriminator(x, wav_add_wm1)
            loss_discriminator0=self.loss_discriminator(wav_judge0,watermark_judge0)
            loss_discriminator1=self.loss_discriminator(wav_judge1,watermark_judge1)
            loss_generator0=self.loss_generator(x, wav_add_wm0, payload, payload_detect0,watermark_judge0)
            loss_generator1=self.loss_generator(x, wav_add_wm1, payload1, payload_detect1,watermark_judge1)
            loss_generator=loss_generator0+loss_generator1
            loss_discriminator=loss_discriminator0+loss_discriminator1

        self.log('val_loss_generator', loss_generator, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_discriminator', loss_discriminator, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss_generator+loss_discriminator, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        



if __name__ == '__main__':
    wav=torch.randn(2,16000)
    model=AW(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8)
    model=model
    model.forward(wav)