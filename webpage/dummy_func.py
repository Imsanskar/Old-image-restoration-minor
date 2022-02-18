from .models import Image
from .serializers import ImageSerializer
import copy,io
from PIL import Image as Photo

from django.core.files.images import ImageFile


def dummy(x):
    print(type(x))
    y=x.convert("L")
    y=x
    return y

def work():
    image1=Image.objects.filter().order_by('-pk')[0] #takes the latest data
    input=image1.image
    im=Photo.open(input)
    print(type(input))
    filename = 'output'+str(image1.id)+'.jpg'
    o_im=dummy(im)
    f=io.BytesIO()
    o_im.save(f,'JPEG')
    outputimage=ImageFile(f,name=filename)
    image1.n_image=outputimage
    image1.save()
