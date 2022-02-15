from django.urls import path
from api import views

urlpatterns=[
    path('',views.ImageDetail.as_view()),
    path('new_image',views.NImageDetail.as_view()),
]