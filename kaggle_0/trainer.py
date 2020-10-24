import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MonetDataset
from dataset import PhotoDataset
from model.networks import Discriminator
from model.networks import Generator
from model.ops import init_weights
from model.ops import toTensor


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.log_path = opt.train_log_path
        self.num_epoch = opt.num_epoch

        # Define networks
        self.netG_a2b = Generator(in_channels=opt.in_channels,
                                  out_channels=opt.out_channels,
                                  filters=opt.filters,
                                  norm=opt.norm).to(self.device)
        self.netG_b2a = Generator(in_channels=opt.in_channels,
                                  out_channels=opt.out_channels,
                                  filters=opt.filters,
                                  norm=opt.norm).to(self.device)

        self.netD_a = Discriminator(in_channels=opt.in_channels,
                                    out_channels=opt.out_channels,
                                    filters=opt.filters,
                                    norm=opt.norm).to(self.device)
        self.netD_b = Discriminator(in_channels=opt.in_channels,
                                    out_channels=opt.out_channels,
                                    filters=opt.filters,
                                    norm=opt.norm).to(self.device)

        # Initialize weights
        init_weights(self.netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(self.netG_b2a, init_type='normal', init_gain=0.02)
        init_weights(self.netD_a, init_type='normal', init_gain=0.02)
        init_weights(self.netD_b, init_type='normal', init_gain=0.02)

        # Define loss functions
        self.cycle_loss_fn = nn.L1Loss().to(self.device)
        self.gan_loss_fn = nn.BCELoss().to(self.device)
        self.ident_loss_fn = nn.L1Loss().to(self.device)

        # Define optimizers for generators and discriminators
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_a2b.parameters(),
                            self.netG_b2a.parameters()),
            lr=opt.lr, betas=(0.5, 0.999))

        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_a.parameters(),
                            self.netD_b.parameters()),
            lr=opt.lr, betas=(0.5, 0.999))

        Photo_dataset = PhotoDataset()
        Monet_dataset = MonetDataset()

        self.Photo_loader = DataLoader(dataset=Photo_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_works)
        self.Monet_loader = DataLoader(dataset=Monet_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_works)

    def train(self):
        # writer = SummaryWriter(log_dir=self.log_path)
        epoch = 0

        for epoch in range(epoch + 1, self.num_epoch + 1):
            self.netG_a2b.train()
            self.netG_b2a.train()
            self.netD_a.train()
            self.netD_b.train()

            for batch, data in enumerate(zip(self.Photo_loader, self.Monet_loader)):
                input_a = toTensor(data[0].to(self.device))  # photo images(a)
                input_b = toTensor(data[1].to(self.device))  # monet images(b)

                # forward netG
                output_b = self.netG_a2b(input_a)
                recon_a = self.netG_b2a(output_b)

                output_a = self.netG_b2a(input_b)
                recon_b = self.netG_a2b(output_a)

                # start backward netD
                self.set_requires_grad([self.netD_a, self.netD_b], True)
                self.optimizer_D.zero_grad()

                # backward netD_a
                pred_real_a = self.netD_a(input_a)
                pred_fake_a = self.netD_a(output_a)

                loss_D_a_real = self.gan_loss_fn(
                    pred_real_a, torch.ones_like(pred_real_a))  # real label 1
                loss_D_a_fake = self.gan_loss_fn(
                    pred_fake_a, torch.zeros_like(pred_fake_a))  # fake label 0
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = self.netD_b(input_b)
                pred_fake_b = self.netD_b(output_b)

                loss_D_b_real = self.gan_loss_fn(
                    pred_real_b, torch.ones_like(pred_real_b))  # real label 1
                loss_D_b_fake = self.gan_loss_fn(
                    pred_fake_b, torch.zeros_like(pred_fake_b))  # fake label 0
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                loss_D = loss_D_a + loss_D_b

                # backward Discriminators
                loss_D.backward(retain_graph=True)
                self.optimizer_D.step()

                # backward netG
                self.set_requires_grad([self.netD_a, self.netD_b], False)
                self.optimizer_G.zero_grad()

                pred_fake_a = self.netD_a(output_a)
                pred_fake_b = self.netD_b(output_b)

                # Gan loss
                loss_G_a2b = self.gan_loss_fn(
                    pred_fake_a, torch.ones_like(pred_fake_a))
                loss_G_b2a = self.gan_loss_fn(
                    pred_fake_b, torch.ones_like(pred_fake_b))

                # Cycle loss
                loss_cycle_a = self.cycle_loss_fn(recon_a, input_a)
                loss_cycle_b = self.cycle_loss_fn(recon_b, input_b)

                # Identity loss
                ident_a = self.netG_b2a(input_a)
                ident_b = self.netG_a2b(input_b)

                loss_ident_a = self.ident_loss_fn(ident_a, input_a)
                loss_ident_b = self.ident_loss_fn(ident_b, input_b)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                    self.opt.wgt_cycle * (loss_cycle_a + loss_cycle_b) + \
                    self.opt.wgt_ident * \
                    (self.opt.wgt_cycle * (loss_ident_a + loss_ident_b))

                # backward Generators
                loss_G.backward()
                self.optimizer_G.step()

                print(f'batch: {batch}, loss_G: {loss_G}')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
