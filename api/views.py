from django.shortcuts import render
from api.models import Image
from rest_framework import generics 
from api.serializers import ImageSerializer
# Create your views here.class SnippetDetail(generics.RetrieveUpdateDestroyAPIView):
    
class ImageDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
