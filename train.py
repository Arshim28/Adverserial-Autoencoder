import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import itertools

from data import train_loader
from model import Encoder, Decoder, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image

config_path = './config.json'
with open(config_path) as config_file:
    config = json.load(config_file)

# Loss function
adverserial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

encoder = Encoder()
decoder = Decoder() 
discriminator = Discriminator() 

optimizer_G = optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=config['lr'], betas=(config['b1'], config['b2'])
)
optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))

Tensor = torch.FloatTensor

def sample_image(n_row, batches_done):
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, config['latent_dim']))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(config['n_epochs']):
    for i, (imgs, _) in enumerate(train_loader):
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adverserial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        optimizer_D.zero_grad()
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        real_loss = adverserial_loss(discriminator(encoded_imgs), valid)
        fake_loss = adverserial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(train_loader) + i
        batches_left = config['n_epochs'] * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, config['n_epochs'], i, len(train_loader), d_loss.item(), g_loss.item(), time_left)
        )


        batches_done = epoch * len(train_loader) + i
        if batches_done % config['sample_interval'] == 0:
            sample_image(n_row=10, batches_done=batches_done)









