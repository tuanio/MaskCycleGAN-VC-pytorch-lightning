import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl
import librosa
import numpy as np
import librosa.display
from mask_cyclegan_vc import PatchDiscriminator, ResnetGenerator
from mask_cyclegan_vc.utils import ImagePool, init_weights, set_requires_grad

class MaskCycleGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # generator pair
        self.genX = ResnetGenerator.Generator()
        self.genY = ResnetGenerator.Generator()
        
        # discriminator pair
        self.disX = PatchDiscriminator.get_model()
        self.disY = PatchDiscriminator.get_model()
        
        # additional discriminator for the second adversarial loss
        self.auxDisX = PatchDiscriminator.get_model()
        self.auxDisY = PatchDiscriminator.get_model()
        
        self.lambda_cycle = 10
        self.lambda_idt = 5
        self.step_stop_idt_loss = 10_000
        self.global_train_steps = 0
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None

        for m in [self.genX, self.genY, self.disX, self.disY, self.auxDisX, self.auxDisY]:
            init_weights(m)

        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optG = Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=0.0002, betas=(0.5, 0.999))
        
        optD = Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters(), self.auxDisX.parameters(), self.auxDisY.parameters()),
            lr=0.0001, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)

    def draw_mag(self, img):
        return librosa.amplitude_to_db(img[0, :1].detach().cpu().numpy(), ref=np.max)

    def audio(self, img):
        x = img[0].detach().cpu()
        mag, phase = x[0], x[1]
        return torch.istft(mag + 1j * phase, n_fft=510, win_length=400, hop_length=160,
                            window=torch.hann_window(window_length=400), normalized=False,
                            onesided=True)
            
    def generator_training_step(self, imgA, maskA, imgB, maskB):        
        """cycle images - using only generator nets"""

        fakeB = self.genX(imgA, maskA)
        cycledA = self.genY(fakeB)
        
        fakeA = self.genY(imgB, maskB)
        cycledB = self.genX(fakeA)
        
        sameB = self.genX(imgB)
        sameA = self.genY(imgA)
        
        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')
        
        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')
        
        # second adversarial loss with auxiliary discriminators
        auxPredFakeB = self.auxDisY(fakeB)
        auxMseGenB = self.get_mse_loss(auxPredFakeB, 'real')
        
        auxPredFakeA = self.auxDisX(fakeA)
        auxMseGenA = self.get_mse_loss(auxPredFakeA, 'real')
        
        if self.global_train_steps <= self.step_stop_idt_loss:
            # compute extra losses
            identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)
        else:
            identityLoss = 0
        
        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        
        # gather all losses
        extraLoss = self.lambda_cycle * cycleLoss + self.lambda_idt * identityLoss
        self.genLoss = mseGenA + mseGenB + auxMseGenA + auxMseGenB + extraLoss
        self.log('gen_loss', self.genLoss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()

        # log imgA - fakeB
        # log imgB - fakeA
        self.logger.log_image("Mag-realA_fakeB", [self.draw_mag(imgA), self.draw_mag(fakeB),])
        self.logger.log_image("Mag-realB_fakeA", [self.draw_mag(imgB), self.draw_mag(fakeA),])

        self.logger.log_audio("Audio-realA_fakeB", [self.audio(imgA), self.audio(fakeB),], sample_rate=[16000, 16000])
        self.logger.log_audio("Audio-realB-fakeA", [self.audio(imgB), self.audio(fakeA),], sample_rate=[16000, 16000])
        
        return self.genLoss
    
    def discriminator_training_step(self, imgA, maskA, imgB, maskB):
        """Update Discriminator"""        
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)
        
        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')
        
        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')
        
        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')
        
        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')
        
        # second adversarial loss with auxiliary discriminators
        auxPredRealA = self.auxDisX(imgA)
        auxMseRealA = self.get_mse_loss(auxPredRealA, 'real')
        
        auxPredFakeA = self.auxDisX(fakeA)
        auxMseFakeA = self.get_mse_loss(auxPredFakeA, 'fake')
        
        auxPredRealB = self.auxDisY(imgB)
        auxMseRealB = self.get_mse_loss(auxPredRealB, 'real')
        
        auxPredFakeB = self.auxDisY(fakeB)
        auxMseFakeB = self.get_mse_loss(auxPredFakeB, 'fake')
        
        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB + auxMseFakeA + auxMseRealA + auxMseFakeB + auxMseRealB)
        self.log('dis_loss', self.disLoss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return self.disLoss
    
    def training_step(self, batch, batch_idx):
        # imgA, imgB = batch['A'], batch['B']
        imgA, maskA, imgB, maskB = batch

        # discriminator_requires_grad = (optimizer_idx==1)
        # set_requires_grad([self.disX, self.disY, self.auxDisX, self.auxDisY], discriminator_requires_grad)
        
        self.global_train_steps += 1

        optG, optD = self.optimizers()

        # schG, schD = self.lr_schedulers()
    
        # Train Generators
        self.toggle_optimizer(optG)
        set_requires_grad([self.disX, self.disY, self.auxDisX, self.auxDisY], False)
        gen_loss = self.generator_training_step(imgA, maskA, imgB, maskB)
        self.manual_backward(gen_loss)
        optG.step()
        optG.zero_grad()
        self.untoggle_optimizer(optG)

        # Train Discriminators
        self.toggle_optimizer(optD)
        set_requires_grad([self.disX, self.disY, self.auxDisX, self.auxDisY], True)
        dis_loss = self.discriminator_training_step(imgA, maskA, imgB, maskB)
        self.manual_backward(dis_loss)
        optD.step()
        optD.zero_grad()
        self.untoggle_optimizer(optD)

        # if optimizer_idx == 0:
        #     return self.generator_training_step(imgA, imgB)
        # else:
        #     return self.discriminator_training_step(imgA, imgB)        
