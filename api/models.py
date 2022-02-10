from django.db import models

from django.utils.translation import gettext_lazy as _

def upload_to(instance,filename):
    return '{filename}'.format(filename=filename)

# Create your models here.
class Image(models.Model):
    image=models.ImageField(_("Image"), upload_to=upload_to, height_field=None, width_field=None, max_length=None)
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')
