#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import cv2
from PIL import Image
import os
from io import BytesIO
from tqdm.notebook import tqdm, trange
import time

def blend_image(original_image: Image.Image, blend_image_source: Image.Image, intensity = 0.4):
    """
        Blends the original image with blend_image with intensity
    """
    
    # converts the blend_image to the format of original image
    # because both the image needs to be in the same format
    # same goes for size
    blend_image_source = blend_image_source.convert(original_image.mode) 
    blend_image_source = blend_image_source.resize(original_image.size)
    
    new_image = Image.new(original_image.mode, original_image.size)
    new_image = Image.blend(original_image, blend_image_source, intensity)
    
    return new_image


# In[3]:


def pil_to_np(img_pil):
    """
        Converts image from pil Image to numpy array
    """
    ar: np.ndarray = np.array(img_pil)
    if len(ar.shape) == 3:
        """
            Tensor transpose, since in this case tensor is 3D the order of transpose can be different
            In 2D matrix the transpose is only i,j-> j,i but in more than 2D matrix different permutation can be 
            applied
        """
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


# In[4]:


def np_to_pil(img_np):
    """
        Converts np.ndarray to Image.Image object
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


# In[5]:


def synthesize_salt_pepper(image: Image.Image, amount, salt_vs_pepper):
    """
        Salt and pepper noise is also known as an impulse noise, this noise can be caused by sharp and sudden 
        disturbances in the image signal. gives the appearance of scattered white or black(or both) pixel over
        the image
    """
    img_pil=pil_to_np(image)

    out = img_pil.copy()
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=img_pil.shape,
                               p=[p, 1 - p])
    salted = np.random.choice([True, False], size=img_pil.shape,
                              p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0.
    noisy = np.clip(out, 0, 1).astype(np.float32)
    return np_to_pil(noisy)


# In[6]:


def synthesize_speckle(image,std_l,std_r):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)


# In[7]:


def blur_image_v2(img):
    x=np.array(img)
    kernel_size_candidate=[(3,3),(5,5),(7,7)]
    kernel_size=random.sample(kernel_size_candidate,1)[0]
    std=random.uniform(1.,5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,std)

    return Image.fromarray(blur.astype(np.uint8))


# In[8]:


def synthesize_low_resolution(image: Image.Image):
    """
        Creates a low resolution image from high resolution image
    """
    width, height = image.size
    
    new_width = np.random.randint(int(width / 2), width - int(width / 5))
    new_height = np.random.randint(int(height / 2), height - int(height / 5))
    
    image = image.resize((new_width, new_height), Image.BICUBIC)
    
    if random.uniform(0, 1) < 0.5:
        image = image.resize((width, height), Image.NEAREST)
    else:
        image = image.resize((width, height), Image.BILINEAR)
        
    return image


# In[9]:


def online_add_degradation_v2(img):
    task_id = np.random.permutation(4)

    for x in task_id:
        if x == 0 and random.uniform(0,1)<0.7:
            img = blur_image_v2(img)
        if x == 1 and random.uniform(0,1)<0.7:
            flag = random.choice([1, 2, 3])
            if flag == 1:
                pass
                # img = synthesize_gaussian(img, 5, 50)
            if flag == 2:
                img = synthesize_speckle(img, 5, 50)
            if flag == 3:
                img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
        if x == 2 and random.uniform(0,1)<0.7:
            img=synthesize_low_resolution(img)

        if x==3 and random.uniform(0,1)<0.7:
            img=convertToJpeg(img,random.randint(40,100))

    return img


# In[10]:


def zero_mask(row, col):
    x = np.zeros((row, col, 3))
    mask=Image.fromarray(x).convert("RGB")
    return mask

def irregular_hole_synthesize(img, mask):
    """
        Create holes using scrach paper textures
        Args:
            img: Original Image
            mask: scratch paper texture
    """
    img_np = np.array(img).astype('uint8')
    mask = mask.resize(img.size)
    mask = mask.convert(img.mode) 
    mask_np = np.array(mask).astype('uint8')
    mask_np = mask_np / 255
    img_new=img_np * (1 - mask_np) + mask_np * 255


    hole_img=Image.fromarray(img_new.astype('uint8')).convert("RGB")


    return hole_img,mask.convert("L")


# In[11]:


def convertToJpeg(im,quality):
    with BytesIO() as f:
        im.save(f, format='JPEG',quality=quality)
        f.seek(0)
        return Image.open(f).convert('RGB')


# In[12]:

