from genericpath import exists
from time import time
from django.http import HttpResponse
from django.shortcuts import render
from .bringing_old_photos_back_to_life import model
import os
import shutil
import torch
from rest_framework.decorators import api_view
from django.shortcuts import render
from .models import Image 
from rest_framework import generics 
from .serializers import ImageSerializer 
from rest_framework.parsers import MultiPartParser, FormParser
from .dummy_func import work    
from rest_framework.response import Response
from rest_framework.views import APIView



from .dummy_func import work

# Create your views here.
def home(request):
	#work()
	# torch.cuda.empty_cache()
	# current_working_directory = os.getcwd().split('/')[-1]
	# path = "webpage/pix2pix/data/train/old_images/001234.jpg"
	# curr_time = time()
	# if not os.path.exists('webpage/input_folder'):
	# 	os.makedirs("webpage/input_folder/")
	# # shutil.copy(path, "webpage/input_folder/input.jpg")
	# # im = Image.open("./pix2pix/data/train/old_images/001000.jpg")
	# model.modify("webpage/input_folder", False)
	# print(f"Time taken: {time() - curr_time}")
	return HttpResponse("Hello there")

class ImageList(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request,*args, **kwargs):
        for x in Image.objects.all():
            x.delete()
        print(request.data)
        images_serializer = ImageSerializer(data=request.data)
        if images_serializer.is_valid():
            images_serializer.save()
            work()
            return Response(images_serializer.data,)
        else:
            print('error', images_serializer.errors)
            return Response(images_serializer.errors)

