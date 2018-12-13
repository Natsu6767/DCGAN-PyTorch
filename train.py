import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from utils import get_celeba
from dcgan import weights_init, Generator, Discriminator

seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

params = {
    "bsize" : 128,
    'imsize' : 64,
    'nc' : 3,
    'nz' : 100,
    'ngf' : 64,
    'ndf' : 64,
    'nepochs' : 5,
    'lr' : 0.0002,
    'beta1' : 0.5}

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

dataloader = get_celeba(params)

sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))

plt.show()

netG = Generator(params).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(params).to(device)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

for epochs in range(params['nepochs']):
	for i, data in enumerate(dataloader, 0):
		# Transfer data tensor to GPU/CPU (device)
		real_data = data.to(device)
		# Get batch size. Can be different from params['nbsize'] for last batch in epoch.
		b_size = real_data.size(0)
        
        # Make accumalated gradients of the discriminator zero.
		netD.zero_grad()
		# Create labels for the real data. (label=1)
		label = torch.full((b_size, ), real_label, device=device)
		output = netD(real_data).view(-1)
		errD_real = criterion(output, label)
		# Calculate gradients for backpropagation.
		errD_real.backward()
		D_x = output.mean().item()
        
        # Sample random data from a unit normal distribution.
		noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
		# Generate fake data (images).
		fake_data = netG(noise)
		# Create labels for fake data. (label=0)
		label.fill(fake_data)
		# Calculate the output of the discriminator of the fake data.
		# As no gradients w.r.t. the generator parameters are to be
		# calculated, detach() is used. Hence, only gradients w.r.t. the
		# discriminator parameters will be calculated.
		# This is done because the loss functions for the discriminator
		# and the generator are slightly different.
		output = netD(fake_data.detach()).view(-1)
		errD_fake = criterion(output, label)
		# Calculate gradients for backpropagation.
		errD_fake.backward()
		D_G_z1 = output.mean().item()

		# Net discriminator loss.
		errD = errD_real + errD_fake
		# Update discriminator parameters.
		optimizerD.step()
        
        # Make accumalted gradients of the generator zero.
		netG.zero_grad()
		# We want the fake data to be classified as real. Hence
		# real_label are used. (label=1)
		label.fill(real_label)
		# No detach() is used here as we want to calculate the gradients w.r.t.
		# the generator this time.
		output = netD(fake_data).view(-1)
		errG = criterion(label, output)
		# Gradients for backpropagation are calculated.
		# Gradients w.r.t. both the generator and the discriminator
		# parameters are calculated, however, the generator's optimizer
		# will only update the parameters of the generator. The discriminator
		# gradients will be set to zero in the next iteration by netD.zero_grad()
		errG.backward()

		D_G_z2 = output.mean().item()
		# Update generator parameters.
		optimizerG.step()

		# Check progress of training.
		if i%100 == 0:
			print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

		# Save the losses for plotting.
		G_losses.append(errG.item())
		D_losses.append(errD.item())

		# Check how the generator is doing by saving G's output on a fixed noise.
		if (iters % 500 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iter += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()