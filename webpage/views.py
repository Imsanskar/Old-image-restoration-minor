from genericpath import exists
from time import time
from django.http import HttpResponse
from django.shortcuts import render
from .bringing_old_photos_back_to_life import model
from PIL import Image
import os
import shutil
import torch

# Create your views here.
def home(request):
	torch.cuda.empty_cache()
	current_working_directory = os.getcwd().split('/')[-1]
	path = "webpage/pix2pix_super_res/outputs/input.jpg"
	curr_time = time()
	if not os.path.exists('webpage/input_folder'):
		os.makedirs("webpage/input_folder/")
	# shutil.copy(path, "webpage/input_folder/input.jpg")
	# im = Image.open("./pix2pix/data/train/old_images/001000.jpg")
	model.modify("webpage/input_folder", True)
	print(f"Time taken: {time() - curr_time}")
	return HttpResponse("Hello there")
