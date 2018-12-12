import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seec(seed)

def weights_init(w):
	classname = w.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(w.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(w.weight.data, 1.0, 0.02)
		nn.init.constant_(w.bias.data, 0)

