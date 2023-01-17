import os
import random
import numpy as np
from argparse import Namespace
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.psp import pSp
from criteria.vgg_loss import VGGLoss
from utils.common import image2tensor, tensor2image, imshow


class EncodeMaster:
	def __init__(self):
		self.root = os.getcwd()
		self.modelPath = f'{self.root}/pretrained_models/psp_ffhq_encode.pt'
		self.device = 'cuda'

		ckpt = torch.load(self.modelPath, map_location='cpu')
		opts = ckpt['opts']
		opts['output_size'] = 1024
		opts['checkpoint_path'] = self.modelPath

		self.net = pSp(Namespace(**opts))
		self.net.eval()
		self.net.to(self.device)
		print('Model loaded')

		for param in self.net.parameters():
			param.requires_grad = False

		batch_size = 1

		transform = transforms.Compose([
			transforms.Resize(1024),
			transforms.CenterCrop(1024),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])

		self.transform256 = transforms.Resize(256)
		self.transform1024 = transforms.Resize(1024)

		self.dataset = datasets.ImageFolder(root=f'{self.root}/images', transform=transform)
		self.loader = iter(torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False))
		self.vgg_loss = VGGLoss(self.device)

	def loadNextImages(self):
		self.imgs1024, _ = next(self.loader)
		return self.transform256(self.imgs1024.to(self.device))

	def _encode(self, image):
		z = self.net.encoder(image)
		z += self.net.latent_avg.repeat(z.shape[0], 1, 1)
		return z

	def _decode(self, z):
		imgs, _ = self.net.decoder([z], 
			input_is_latent=True,
			randomize_noise=False)
		return imgs

	def getInitialProjection(self, image):
		z0 = self._encode(image)
		imgs = self._decode(z0)

		self.imgs_real = torch.cat([img for img in self.imgs1024], dim=1)
		imgs_fakes = torch.cat([img for img in imgs], dim=1)

		print('initial projections:')
		imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)), 10)

		return z0

	def train(self, z0, iters=500, showEvery=100):
		z = z0.detach().clone()
		z.requires_grad = True
		optimizer = torch.optim.Adam([z], lr=0.01)

		for step in range(iters):
			imgs_gen = self._decode(z)
			imgs_gen256 = self.transform256(imgs_gen)
			z_hat = self._encode(imgs_gen256)

			loss = F.mse_loss(imgs_gen, self.imgs1024) + self.vgg_loss(imgs_gen, self.imgs1024) + F.mse_loss(z0, z_hat)*2.0

			optimizer.zero_grad()
			loss.backward()
			optimizer.step() 

			if (step+1) % showEvery == 0:
				print(f'step: {step+1}, loss: {np.round(loss.item(), 3)}')
				imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)        
				imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)),10)

	def saveLatent(self, name, z):
		np.save(f'{self.root}/saved_latents/{name}.npy', z.detach().squeeze().cpu().numpy())

