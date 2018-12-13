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

seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

dataloader = get_celeba(params)

state_dict = torch.load(args.load_path)

netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
print(netG)

params = state_dict['params']
noise = torch.randn(1, params['nz'], 1, 1)

with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

# Plot the fake image.
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Generated Images")
plt.imshow(generated_img)
plt.show()