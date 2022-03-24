import shutil
from cv2 import transform
import torch

from webpage import MODEL_PATH_GENERATOR, MODEL_PATH_GENERATOR_TRANSPOSE
from .models import Image
from .serializers import ImageSerializer
import copy,io
from PIL import Image as Photo
from django.core.files.images import ImageFile
import os
from .bringing_old_photos_back_to_life import model
import glob
from .pix2pix_transpose import model as  model_transpose
from .pix2pix import model as pix2pix_model
from torchvision import transforms
from .pix2pix.data.image_manipulation import np_to_pil

def get_file_name(file_path_list:str):
	file_path_list = list(reversed(list(file_path_list)))

	file_name = list(reversed(file_path_list))
	for i, char in enumerate(file_path_list):
		if char == '/' or char == '\\':
			file_name = list(reversed(file_path_list[:i]))

			return "".join(file_name).split(".")[0]

	return "".join(file_name).split(".")[0]


def restore_image(image_file:str, method):
	file_name = get_file_name(image_file)
	if os.path.exists("webpage/input_folder/output.png"):
		os.remove("webpage/input_folder/output.png")
	files = glob.glob('webpage/input_folder/*', recursive=True)
	for f in files:
		if get_file_name(f) != file_name:
			print("LOG: Old files removed")
			os.remove(f)
	if method == 3:
		torch.cuda.empty_cache()
		# cleanup intermediate directory
		# os.remove(image_file)
	
		files = glob.glob('webpage/output_folder/final_output/*', recursive=True)
		for f in files:
			os.remove(f)

		shutil.rmtree("webpage/output_folder/final_output/", ignore_errors=True)
		shutil.rmtree("webpage/output_folder/stage_1_restore_output/input_image/", ignore_errors=True)
		shutil.rmtree("webpage/output_folder/stage_1_restore_output/origin/", ignore_errors=True)
		shutil.rmtree("webpage/output_folder/stage_1_restore_output/restored_image/", ignore_errors=True)
		shutil.rmtree("webpage/output_folder/stage_1_restore_output/masks/input/", ignore_errors=True)
		shutil.rmtree("webpage/output_folder/stage_1_restore_output/masks/mask/", ignore_errors=True)

		model.modify("webpage/input_folder", with_scratch=True, image_filename = image_file)
		y = Photo.open(f"webpage/output_folder/stage_1_restore_output/restored_image/{file_name}.png")
		return y
	elif method == 1:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		generator_transpose = model_transpose.get_generator(MODEL_PATH_GENERATOR_TRANSPOSE).to(device)

		image = Photo.open(image_file)
		tensor_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((512, 512))
		]) 

		# apply model to the input
		# unsqueeze to add new dimension the model takes N*channel*image_width*image_height
		# where N is the number of datas
		output_image = generator_transpose(tensor_transform(image).unsqueeze(0).to(device))
		output_image = np_to_pil(
			output_image.detach().cpu().numpy()[0]
		)

		
		return output_image
		

	else:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# apply model to the input
		# unsqueeze to add new dimension the model takes N*channel*image_width*image_height
		# where N is the number of datas
		generator = pix2pix_model.get_generator(MODEL_PATH_GENERATOR).to(device)
		image = Photo.open(image_file)
		tensor_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((512, 512))
		]) 

		output_image = generator(tensor_transform(image).unsqueeze(0).to(device))
		

		output_image = np_to_pil(
			output_image.detach().cpu().numpy()[0]
		)

		
		return output_image
		


# restores the image from input_folder and saves it in the database
# method describes which method to use for restoration
def convert(method):
	image1 = Image.objects.filter().order_by('-pk')[0] #takes the latest data
	input = image1.image
	method = image1.method
	im = Photo.open(input)
	filename  =  "output.png"
	o_im = restore_image(image1.image.path, method=method)
	f = io.BytesIO()
	o_im.save(f,'PNG')
	outputimage = ImageFile(f,name = filename)
	image1.n_image = outputimage
	image1.save()

	
	# cleanup useless files
	files = os.listdir('./webpage/input_folder/')
	for f in files:
		if f != "output.png" and f != 'Default.jpg':
			print(f"Fuse file removed {f}")
			os.remove(f"webpage/input_folder/{f}")
