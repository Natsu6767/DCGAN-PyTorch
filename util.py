import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

root = 'data/celeba'

def get_celeba(params):
	transforms = transforms.Compose([
		transforms.Resize(params['imsize']),
		transforms.CenterCrop(params['imsize']),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5),
			(0.5, 0.5, 0.5))])

	dataset = dset.ImageFolder(root=root, transforms=transforms)

	dataloader = torch.utils.data.DataLoader(dataset,
		batch_size=params['bsize'],
		shuffle=True)

	return dataloader