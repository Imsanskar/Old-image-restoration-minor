from django.urls import path
from api import views

urlpatterns=[
    path('<int:pk>/',views.ImageDetail.as_view()),
    path('',views.ImageList.as_view()),
    path('new_image',views.NImageDetail.as_view()),
]