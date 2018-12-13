import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

# Get latent vector Z from unit normal distribution.
noise = torch.randn(1, params['nz'], 1, 1, device=device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
	# Get generated image from the noise vector using
	# the trained generator.
    generated_img = netG(noise).detach().cpu()

# Display the generated image.
plt.title("Generated Images")
plt.imshow(np.transpose(generated_img[-1], (1,2,0)))
plt.show()