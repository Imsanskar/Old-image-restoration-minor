from pyrsistent import b
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
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


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


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.0):
        super(UNetUp, self).__init__()

        layers = [
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_size, out_size, in_size // 2),
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


import torch.nn.functional as F
import math
def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret



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

def calculate_fid(model, images_1, images_2):
	preprocess = transforms.Compose([
		transforms.Resize(299),
#         transforms.CenterCrop(299),
#         transforms.ToTensor(),
	])
	images_1 = preprocess(images_1)
	images_2 = preprocess(images_2)
	img1_vec = model(preprocess(images_1)).detach().cpu().numpy()
	img2_vec = model(preprocess(images_2)).detach().cpu().numpy()
	return fid(img1_vec, img2_vec)


if __name__ == "__main__":
	# batch size
	batch_size = 10

	# image size
	image_height = 512
	image_width = 512


	#initialize model classes
	generator = GeneratorUNet()
	discriminator = Discriminator()


	# check if cuda is avialbale
	cuda = True if torch.cuda.is_available() else False
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	generator_file = "saved_models_new_data/generator_131.pth"
	# initialize weights if the model is not found in the paths
	if os.path.exists(generator_file):
		print("Generator Found")
		generator.load_state_dict(torch.load(generator_file, map_location = device))
	else:
		generator.apply(weights_init_normal)

	

	# to cuda if cuda is avaiable
	generator.to(device)
	
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
		batch_size = 2,
		shuffle = True,
	)

	val_image_dataloader = img_dataloader.ImageDataset("./data/val/old_image", "./data/val/reconstructed_image", transform)
	val_dataloader = DataLoader(
		val_image_dataloader,
		batch_size = 2,
		shuffle = True
	)

	# FID calculation
	# preprocess = transforms.Compose([
	# 		transforms.Resize(299),
	# 	])
	
	# model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True, progress=False)
	# model.eval()
	# model.to(device)

	# generator.eval()
	# # score for training set
	# dataloader_list = next(iter(dataloader))
	# images_1 = preprocess(generator(dataloader_list['A'].to(device)))
	# images_2 = preprocess(dataloader_list['B']).to(device)

	# # score of the dataser
	# img1_vec = model(preprocess(dataloader_list['A'].to(device))).detach().cpu().numpy()
	# img2_vec = model(preprocess(images_2)).detach().cpu().numpy()
	# print(f"Dataset score: {fid(np.transpose(img1_vec), np.transpose(img2_vec))}")


	# img1_vec = model(preprocess(images_1)).detach().cpu().numpy()
	# img2_vec = model(preprocess(images_2)).detach().cpu().numpy()
	# print(f"Training score: {fid(np.transpose(img1_vec), np.transpose(img2_vec))}")
	# torch.cuda.empty_cache()

	# dataloader_list = next(iter(val_dataloader))
	# images_1 = preprocess(generator(dataloader_list['A'].to(device)))
	# images_2 = preprocess(dataloader_list['B']).to(device)
	# img1_vec = model(preprocess(images_1)).detach().cpu().numpy()
	# img2_vec = model(preprocess(images_2)).detach().cpu().numpy()
	# print(f"Validation score: {fid(np.transpose(img1_vec), np.transpose(img2_vec))}")

	# with torch.no_grad():
	# 	for i, batch in enumerate(dataloader):
	# 		generate_image  = list(iter(generator(batch['A'].to(device))))
	# 		original_image_list = list(iter(batch['A']))
	# 		pil_transform = transforms.Compose([
	# 			transforms.ToPILImage()
	# 		])

	# 		pil_transform(generate_image[0].detach().cpu()).save(f"outputs/{2 * i}.jpg")
	# 		pil_transform(generate_image[1].detach().cpu()).save(f"outputs/{2 * i + 1}.jpg")
			
	# 		pil_transform(original_image_list[0].detach().cpu()).save(f"inputs/{2 * i}.jpg")
	# 		pil_transform(original_image_list[1].detach().cpu()).save(f"inputs/{2 * i + 1}.jpg")

	# 		if i > 10:
	# 			break

	ssim_value_list = []
	ssim_value_list_dataset = []
	with torch.no_grad():
		for i , batch in enumerate(dataloader):
			generated_image = generator(batch['A'].to(device))
			original_image = batch['B'].to(device)

			ssim_value_list.append(ssim(original_image, generated_image, 255).detach().cpu().item())
			ssim_value_list_dataset.append(ssim(batch['A'].to(device), original_image, 255).detach().cpu().item())


			if i > 100:
				break

	ssim_value_np = np.array(ssim_value_list)
	ssim_value_dataset_np = np.array(ssim_value_list_dataset)

	print(f"SSIM Value: {np.mean((ssim_value_list))}")



	