import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seec(seed)

def weights_init(w):
	classname = w.__class__.__name__
	if classname.find('conv') != -1:
		nn.init.normal_(w.weight.data, 0.0, 0.02)
	elif classname.find('bn') != -1:
		nn.init.normal_(w.weight.data, 1.0, 0.02)
		nn.init.constant_(w.bias.data, 0)

class Generator(nn.Module):
	def __init__(self, params):
		super().__init__()

		self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
			kernel_size=4, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(params['ngf']*8)

		self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
			4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(params['ngf']*4)

		self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
			4, 2, 1, bias=False)
		self.bn3 = bb.BatchNorm2d(params['ngf']*2)

		self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
			4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(params['ngf'])

		self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
			4, 2, 1, bias=False)

	def forward(self, x):
		x = F.relu(self.bn1(self.tconv1(x)))
		x = F.relu(self.bn2(self.tconv2(x)))
		x = F.relu(self.bn3(self.tconv3(x)))
		x = F.relu(self.bn4(self.tconv4(x)))

		x = f.tanh(self.tconv5(x))

		return x
