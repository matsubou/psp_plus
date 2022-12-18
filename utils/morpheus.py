from glob import glob


class Morpheus:
	def __init__(self):
		self.latent = 0
		self.metadata = {}
		self.face_latents = self._load('faces')
		self.directions = self._load('directions')
		self.face_imgs = {}


	def _load(what='faces'):
		if what not in ['faces', 'directions', 'images']:
			raise Exception('Invalid input')

		xs = glob(f'./latents/{what}/*.npy')
		X = {}

		for x in xs:
			name = d.split('.')[-2].split('/')[-1]
			X[name] = np.load(f'{name}.npy')
		return X


	def show_image(image):
		pass


	def ls(showImg=False):
		if showImg:
			if len(self.face_imgs.keys()) == 0:
				self.face_imgs = self._load('images')

			for name, img in list(self.face_imgs.items()):
				print(f'Face: {name}')
				show_image(img)

		else:	
			print(f'Face latents = {list(self.face_latents.keys())}')


	def decode(z):
		# z.shape (18, 512)
		pass


	def interp2(z1, z2, mag):
		self.latent = mag*z1 + (1 - mag)*z2
		self.metadata = {
			'action': 'interp'
			'latents': [z1, z2]
			'mags': [mag, (1 - mag)]
		}

		# decode(self.latent)

