from rest_framework.decorators import api_view
from django.shortcuts import render
from api.models import Image , NImage
from rest_framework import generics 
from api.serializers import ImageSerializer ,NImageSerializer
from api.dummy_func import work    
from rest_framework.response import Response

class ImageDetail(generics.ListCreateAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer

class ImageList(generics.ListAPIView):
    queryset=Image.objects.all()
    serializer_class = ImageSerializer

# class NImageDetail(generics.ListAPIView):
#     queryset = Image.objects.all()
#     serializer_class = NImageSerializer

# @api_view(['GET','DELETE'])
# def NImageDetail(request,format=None):
#     work()
#     if request.method == 'GET':
#         nimage = NImage.objects.all()
#         serializer = NImageSerializer(nimage)  
#         return Response(serializer.data)
#     elif request.method == 'DELETE':
#         nimage =NImage.objects.get(pk=1)
#         nimage.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)
class NImageDetail(generics.ListCreateAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
