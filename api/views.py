from rest_framework.decorators import api_view
from django.shortcuts import render
from api.models import Image , NImage
from rest_framework import generics 
from api.serializers import ImageSerializer ,NImageSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from api.dummy_func import work    
from rest_framework.response import Response
from rest_framework.views import APIView

# class ImageDetail(generics.RetrieveUpdateAPIView):
#     queryset = Image.objects.all()
#     serializer_class = ImageSerializer

# class ImageList(generics.RetriveDestroyAPIView):
#     queryset=Image.objects.all()
#     serializer_class = ImageSerializer


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
# class NImageDetail(generics.ListCreateAPIView):
#     queryset = NImage.objects.all()
#     serializer_class = NImageSerializer
# class NImageList(generics.ListAPIView):
#     queryset = NImage.objects.all()
#     serializer_class = NImageSerializer


class ImageList(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request,*args, **kwargs):
        print(request.data)
        images_serializer = ImageSerializer(data=request.data)
        if images_serializer.is_valid():
            images_serializer.save()
            return Response(images_serializer.data,)
        else:
            print('error', images_serializer.errors)
            return Response(images_serializer.errors)

# class NImageList(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def get(self, request, *args, **kwargs):
#         queryset = NImage.objects.all()
#         serializer = PostSerializer(posts, many=True)
#         return Response(serializer.data)

#     def post(self, request, *args, **kwargs):
#         posts_serializer = PostSerializer(data=request.data)
#         if posts_serializer.is_valid():
#             posts_serializer.save()
#             return Response(posts_serializer.data, status=status.HTTP_201_CREATED)
#         else:
#             print('error', posts_serializer.errors)
#             return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)