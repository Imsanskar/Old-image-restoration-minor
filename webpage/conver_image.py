import shutil
import torch
from .models import Image
from .serializers import ImageSerializer
import copy,io
from PIL import Image as Photo
from django.core.files.images import ImageFile
import os
from .bringing_old_photos_back_to_life import model
import glob

def get_file_name(file_path_list:str):
	file_path_list = list(reversed(list(file_path_list)))

	file_name = list(reversed(file_path_list))
	for i, char in enumerate(file_path_list):
		if char == '/' or char == '\\':
			file_name = list(reversed(file_path_list[:i]))

			return "".join(file_name).split(".")[0]

	return "".join(file_name).split(".")[0]



def restore_image(x, image_file:str):
	torch.cuda.empty_cache()
	# cleanup intermediate directory
	# os.remove(image_file)
	file_name = get_file_name(image_file)
	if os.path.exists("webpage/input_folder/output.png"):
		os.remove("webpage/input_folder/output.png")
	
	files = glob.glob('webpage/input_folder/*', recursive=True)
	for f in files:
		if get_file_name(f) != file_name:
			print("LOG: Old files removed")
			os.remove(f)
	files = glob.glob('webpage/output_folder/final_output/*', recursive=True)
	for f in files:
		os.remove(f)

	shutil.rmtree("webpage/output_folder/final_output/", ignore_errors=True)
	shutil.rmtree("webpage/output_folder/stage_1_restore_output/input_image/", ignore_errors=True)
	shutil.rmtree("webpage/output_folder/stage_1_restore_output/origin/", ignore_errors=True)
	shutil.rmtree("webpage/output_folder/stage_1_restore_output/restored_image/", ignore_errors=True)
	shutil.rmtree("webpage/output_folder/stage_1_restore_output/masks/input/", ignore_errors=True)
	shutil.rmtree("webpage/output_folder/stage_1_restore_output/masks/mask/", ignore_errors=True)

	model.modify("webpage/input_folder", True, image_filename = image_file)
	y = Photo.open(f"webpage/output_folder/stage_1_restore_output/restored_image/{file_name}.png")
	# files = glob.glob('webpage/input_folder/*', recursive=True)
	# for f in files:
	# 	os.remove(f)
	return y

# restores the image from input_folder and saves it in the database
def convert():
	image1 = Image.objects.filter().order_by('-pk')[0] #takes the latest data
	input = image1.image
	method = image1.method
	im = Photo.open(input)
	filename  =  'output.png'
	o_im = restore_image(im, image1.image.path)
	f = io.BytesIO()
	o_im.save(f,'PNG')
	outputimage = ImageFile(f,name = filename)
	image1.n_image = outputimage
	image1.save()

	
	files = os.listdir('./webpage/input_folder/')
	for f in files:
		if f != "output.png" or 'Default.jpg':
			print(f"Fuse file removed {f}")
			os.remove(f"webpage/input_folder/{f}")
