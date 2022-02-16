import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.datasets as dset
from data import image_manipulation
from data import dataloader as img_dataloader
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

torch.cuda.is_available()


# random seed for reproducibility
random_seed = 69

np.random.seed(random_seed)


# no of workers for dataloader
no_of_workers = 4

# root of the data
data_root = "data/train/"

# batch size
batch_size = 1

#no of epochs
n_epochs = 10

# learning rate
lr = 0.0002

# betas for adam
beta_1 = 0.5
beta_2 = 0.999

# image size
image_height = 512
image_width = 512


def weights_init_normal(m):
	classname = m.__class__.__name__

	if classname.find("Conv") != -1 and classname.find("DoubleConv") == 1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
	def __init__(self, in_size, out_size, normalize = True, dropout = 0.0):
		super(UNetDown, self).__init__()
		layers = [
			nn.Conv2d(in_size, out_size, 4, 2, 1, bias = False)
		]
		if normalize:
			layers.append(nn.InstanceNorm2d(out_size))
			
		layers.append(nn.LeakyReLU(0.2))
		
		if dropout:
			layers.append(nn.Dropout(dropout))
			
		self.model = nn.Sequential(*layers)
		
	def forward(self, x):
		return self.model(x)


class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class UNetUp_UpSample(nn.Module):
	def __init__(self, in_size, out_size, dropout = 0.0):
		super(UNetUp, self).__init__()

		layers = [
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			DoubleConv(in_size, out_size, in_size // 2),
		]
		if dropout:
			layers.append(nn.Dropout(dropout))

		self.model = nn.Sequential(*layers)
		
	
	def forward(self, x, skip_input):
		x = self.model(x)
		x = torch.cat((x, skip_input), 1)

		return x

class UNetUp(nn.Module):
	def __init__(self, in_size, out_size, dropout = 0.0):
		super(UNetUp, self).__init__()

		layers = [
			nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(out_size),
			nn.ReLU(inplace=True),
		]
		if dropout:
			layers.append(nn.Dropout(dropout))

		self.model = nn.Sequential(*layers)
		
	
	def forward(self, x, skip_input):
		x = self.model(x)
		x = torch.cat((x, skip_input), 1)

		return x


class GeneratorUNet(nn.Module):
	def __init__(self, in_channels=3, out_channels=3):
		super(GeneratorUNet, self).__init__()
		
		self.down1 = UNetDown(in_channels, 64, normalize=False)
		self.down2 = UNetDown(64, 128)
		self.down3 = UNetDown(128, 256)
		self.down4 = UNetDown(256, 512, dropout=0.5)
		self.down5 = UNetDown(512, 512, dropout=0.5)
		self.down6 = UNetDown(512, 512, dropout=0.5)
		self.down7 = UNetDown(512, 512, dropout=0.5)
		self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

		self.up1 = UNetUp(512, 512, dropout=0.5)
		self.up2 = UNetUp(1024, 512, dropout=0.5)
		self.up3 = UNetUp(1024, 512, dropout=0.5)
		self.up4 = UNetUp(1024, 512, dropout=0.5)
		self.up5 = UNetUp(1024, 256)
		self.up6 = UNetUp(512, 128)
		self.up7 = UNetUp(256, 64)

		self.final = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(128, out_channels, 4, padding=1),
			nn.Tanh(),
		)

	def forward(self, x):
		# U-Net generator with skip connections from encoder to decoder
		d1 = self.down1(x)
		d2 = self.down2(d1)
		d3 = self.down3(d2)
		d4 = self.down4(d3)
		d5 = self.down5(d4)
		d6 = self.down6(d5)
		d7 = self.down7(d6)
		d8 = self.down8(d7)
		
		# unet connections
		u1 = self.up1(d8, d7)
		u2 = self.up2(u1, d6)
		u3 = self.up3(u2, d5)
		u4 = self.up4(u3, d4)
		u5 = self.up5(u4, d3)
		u6 = self.up6(u5, d2)
		u7 = self.up7(u6, d1)

		return self.final(u7)


class Discriminator(nn.Module):
	def __init__(self, in_channels=3):
		super(Discriminator, self).__init__()

		def discriminator_block(in_filters, out_filters, normalization=True):
			"""Returns downsampling layers of each discriminator block"""
			layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
			if normalization:
				layers.append(nn.InstanceNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*discriminator_block(in_channels * 2, 64, normalization=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(512, 1, 4, padding=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, img_A, img_B):
		# Concatenate image and condition image by channels to produce input
		img_input = torch.cat((img_A, img_B), 1)
		return self.model(img_input)


import requests
from io import BytesIO
transform = transforms.Compose([
	transforms.ToTensor(), # transform to tensor
	transforms.Resize((image_width, image_height)) # Resize the image to constant size
])




# count the number of trainable parameters
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)




if __name__ == "__main__":
	# no of workers for dataloader
	no_of_workers = 4

	# root of the data
	data_root = "data/train/"

	# batch size
	batch_size = 10

	#no of epochs
	n_epochs = 10

	# learning rate
	lr = 0.0002

	# betas for adam
	beta_1 = 0.5
	beta_2 = 0.999

	# image size
	image_height = 512
	image_width = 512


	#initialize model classes
	generator = GeneratorUNet()
	discriminator = Discriminator()


	# check if cuda is avialbale
	cuda = True if torch.cuda.is_available() else False
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(cuda)

	generator_file = "saved_models/generator_new.pth"
	discriminator_file = "saved_models/discriminator_new.pth"
	# initialize weights if the model is not found in the paths
	if os.path.exists(generator_file):
		print("Generator Found")
		generator.load_state_dict(torch.load(generator_file, map_location = device))
	else:
		generator.apply(weights_init_normal)

	
	if os.path.exists(discriminator_file):
		print("Discriminator Found")
		discriminator.load_state_dict(torch.load(discriminator_file, map_location = device))
	else:
		discriminator.apply(weights_init_normal)

	# model loss functions
	loss_fn_generator = torch.nn.MSELoss() # mean squared loss
	loss_fn_disc = torch.nn.L1Loss() #pixel wise loss

	# to cuda if cuda is avaiable
	generator.to(device)
	discriminator.to(device)
	loss_fn_disc.to(device)
	loss_fn_generator.to(device)
	
	# optimizers
	optimier_G = torch.optim.Adam(generator.parameters(), betas=(beta_1, beta_2), lr=lr)
	optimier_D = torch.optim.Adam(discriminator.parameters(), betas=(beta_1, beta_2), lr=lr)

	# Loss weight of L1 pixel-wise loss between translated image and real image
	lambda_pixel = 100

	# Calculate output of image discriminator (PatchGAN)
	patch = (1, image_height // 2 ** 4, image_width // 2 ** 4)

	# Tensor type
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




	transform = transforms.Compose([
		transforms.ToTensor(), # transform to tensor
		transforms.Resize((image_width, image_height)) # Resize the image to constant size
	])

	# create a dataloader
	pair_image_dataloader = img_dataloader.ImageDataset("./data/train/old_images", "./data/train/reconstructed_images", transform)

	dataloader = DataLoader(
		pair_image_dataloader,
		batch_size = 4,
		shuffle = True,
	)

	# val_image_dataloader = img_dataloader.ImageDataset("./data/val/old_image", "./data/val/reconstructed_image", transform)
	# val_dataloader = DataLoader(
	#     val_image_dataloader,
	#     batch_size = 5,
	#     shuffle = True
	# )

	generator_error_file = "generator.csv"
	disc_error_file = "discriminator.csv"

	total_generator_epochs = 0
	try: 
		with open(generator_error_file, "r") as f:
			last_line = None
			for last_line in f:
				pass
		if last_line != None:
			print("CSV file found")
			total_generator_epochs = int(last_line.split(',')[-1]) + 1
		else:
			total_generator_epochs = 0

		f.close()
	except FileNotFoundError:
		with open(generator_error_file, "w") as f:
			total_generator_epochs = 0

		f.close()


	total_discriminator_epochs = 0
	try: 
		with open(disc_error_file, "r") as f:
			last_line = None
			for last_line in f:
				pass
		if last_line != None:
			print("CSV file found")
			total_discriminator_epochs = int(last_line.split(',')[-1]) + 1
		else:
			total_discriminator_epochs = 0

		f.close()
	except FileNotFoundError:
		with open(disc_error_file, "w") as f:
			total_generator_epochs = 0

		f.close()

	for epoch in range(1):
		print(discriminator)
		# break
		loss_G_list = np.array([])
		loss_D_list = np.array([])
		for i, batch in tqdm(enumerate(dataloader)):
			real_A = batch['A'] # old image
			real_B = batch['B'] # new image
	
			# train generator
			optimier_G.zero_grad()
	
			 # Adversarial ground truths
			valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False) # ground truth for valid
			fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False) # ground truth for invalid
	
	
			# GAN loss
			fake_B = generator(real_A.to(device)) # fake sample generated by generator
			pred_fake = discriminator(fake_B.to(device), real_B.to(device)) # prediction using discriminator
			loss_generator = loss_fn_generator(pred_fake.to(device), valid.to(device)) # check if the sample is valid or not
	
			loss_pixel = loss_fn_disc(fake_B.to(device), real_B.to(device)) # calculate the pixel wise loss
	
			# total loss
			loss_G = loss_generator + lambda_pixel * loss_pixel # total loss of the generator
	
			loss_G.backward()
			optimier_G.step()
	
			## Train discriminator
			optimier_D.zero_grad()
	
			# Real loss
			pred_real = discriminator(real_B.to(device), real_A.to(device)) # loss to check real or not
			loss_real = loss_fn_generator(pred_real, valid)

			# Fake loss
			pred_fake = discriminator(fake_B.detach().to(device), real_A.to(device)) # loss to check fake or not
			loss_fake = loss_fn_generator(pred_fake.to(device), fake.to(device))

			# Total loss
			loss_D = 0.5 * (loss_real + loss_fake) # total loss of the discriminator
	
			loss_D.backward()
			optimier_D.step()
	
			# for logging
			if i % 10 == 0 and i != 0:
				print(f"Generator Error: {torch.linalg.norm(loss_G).item()}, epoch: {epoch}, itr: {i}")
				print(f"Discriminator Error: {torch.linalg.norm(loss_D).item()}, epoch: {epoch}, itr: {i}")
	
			loss_G_list = np.append(loss_G_list,  torch.linalg.norm(loss_G).item())
			loss_D_list = np.append(loss_D_list,  torch.linalg.norm(loss_D).item())
	
		# log into a file
		with open(generator_error_file, "a") as f:
			f.write(f"Generator Error, {torch.linalg.norm(loss_G).item()}, {epoch + total_generator_epochs}\n")
		with open(disc_error_file, "a") as f:
			f.write(f"Discriminator Error, {torch.linalg.norm(loss_D).item()}, {epoch + total_discriminator_epochs}\n")
	
		torch.save(generator.state_dict(), f"saved_models/generator_new.pth")
		torch.save(discriminator.state_dict(), f"saved_models/discriminator_new.pth")
	
		torch.cuda.empty_cache()

		# image load
		original_image = Image.open("./outputs/input.jpg")
		# original_image = test_list[0]["A"]
		original_image.save("outputs/input.jpg")
		output_image = image_manipulation.np_to_pil(
			generator(transform(original_image).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
		)
		
		width, height = original_image.size
		output_image.resize((width, height))

		output_image.save(f"outputs/output_{total_generator_epochs}.jpg")




from scipy.linalg import sqrtm
def fid(img1_vec, img2_vec) -> float:
	#calculate mean
	mu1, C1 = img1_vec.mean(axis = 0), np.cov(img1_vec, rowvar = False)
	mu2, C2 = img2_vec.mean(axis = 0), np.cov(img2_vec, rowvar = False)

	# sum of squared difference
	msdiff = np.sum((mu1 - mu2) ** 2)

	# sqrt of products
	product_covariance = sqrtm(C1.dot(C2)) 
	if np.iscomplexobj(product_covariance):
		product_covariance = product_covariance.real

	sqrt_product_covariance = np.trace(C1 + C2 - 2 * product_covariance)
	#return the result
	return msdiff + sqrt_product_covariance



