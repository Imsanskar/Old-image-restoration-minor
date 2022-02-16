from rest_framework.decorators import api_view
from django.shortcuts import render
from api.models import Image , NImage
from rest_framework import generics 
from api.serializers import ImageSerializer ,NImageSerializer
from api.dummy_func import work    
from rest_framework.response import Response

class ImageDetail(generics.RetrieveUpdateAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer

class ImageList(generics.ListAPIView):
    queryset=Image.objects.all()
    serializer_class = ImageSerializer


class NImageList(generics.ListCreateAPIView):
    queryset = NImage.objects.all()
    serializer_class = NImageSerializer

class NImageDetail(generics.RetrieveUpdateAPIView):
    queryset = NImage.objects.all()
    serializer_class = NImageSerializer
