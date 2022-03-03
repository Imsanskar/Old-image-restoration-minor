from rest_framework import serializers
from api.models import Image ,NImage

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model=Image
        fields=['id','image']

class NImageSerializer(serializers.ModelSerializer):
    class Meta:
        model=NImage
        fields=['nimage']