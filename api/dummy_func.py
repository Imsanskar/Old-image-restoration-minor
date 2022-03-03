from api.models import Image,NImage
from api.serializers import ImageSerializer,NImageSerializer
import copy
from PIL import Image as Photo

def dummy(x):
    y=copy.deepcopy(x)
    return y

def work():
    image1=Image.objects.filter()[0] #takes the latest data
    input=image1.image
    output=dummy(input)
    a = NImage(nimage=output)
    a.save()
    image2=NImage.objects.all()[0]
    output=image2.nimage
    # print(image2)
    im = Photo.open(output)
   # im.show(output)
