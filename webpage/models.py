from django.db import models
from django.utils.translation import gettext_lazy as _


def upload_to(instance,filename):
    return '{filename}'.format(filename=filename)

class Image(models.Model):
    image = models.ImageField(_("Image"), upload_to=upload_to, height_field=None, width_field=None, max_length=None)
    n_image = models.ImageField("New_Image",default='Default.jpg')
