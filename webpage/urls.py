from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),

    path('api/old_image/',views.ImageList.as_view()),
    
]