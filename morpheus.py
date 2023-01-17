# import sys
# sys.path.append('.')
# sys.path.append('..')
import os
import numpy as np
from argparse import Namespace
from models.psp import pSp
import torch
from utils.common import tensor2image, imshow


class Morpheus:
	def __init__(self):
		self.root = 'D:/PythonProjects/psp_plus' 
		self.imgsPath = f'{self.root}/images'
		self.latentPath = f'{self.root}/latents/faces'
		self.directionPath = f'{self.root}/latents/directions'
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

		self.image = None
		self.latent = None
		self.dirs = {}
		self.loadLatentDirections()

	def loadLatent(self, filename):
		z = np.load(f'{self.latentPath}/{filename}.npy')
		self.latent = torch.from_numpy(z).to(self.device)
		return self.latent

	def loadLatentDirections(self):
		dirs = os.listdir(self.directionPath)
		for d in dirs:
			name = d.split('.')[-2]
			z = np.load(f'{self.directionPath}/{name}.npy')			
			self.dirs[name] = torch.from_numpy(z).to(self.device)

	def decode(self, latent=None):
		if latent is not None:
			if len(latent.shape) < 3:
				z = torch.unsqueeze(latent, dim=0)

			img, _ = self.net.decoder(
					[z],
					input_is_latent=True,
					randomize_noise=False
				)
			self.image = tensor2image(img[0])
			return self.image
		else:
			raise Exception('No latent loaded')

	def showImg(self):
		if self.image is not None:
			imshow(self.image)

	def morph(self, directions, mags):
		assert len(directions) == len(mags)
		temp = torch.clone(self.latent)
		for m, d in zip(mags, directions):
			temp += m*self.dirs[d]
		self.decode(latent=temp)

	def interp2(self, z1, z2, mag=0.5):
		temp = mag*z1 + (1-mag)*z2
		self.decode(latent=temp)

	def interp2Panorama(z1, z2, magRange=[0 1], imgs=5):
		

