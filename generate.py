import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from utils import get_celeba
from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
args = parser.parse_args()

state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']

dataloader = get_celeba(params)

netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
print(netG)

noise = torch.randn(1, params['nz'], 1, 1, device=device)

with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

print(generated_img.shape)

# Plot the fake image.
plt.title("Generated Images")
plt.imshow(np.transpose(generated_img[-1], (1,2,0)))
plt.show()