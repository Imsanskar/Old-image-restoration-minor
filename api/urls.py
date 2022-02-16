from django.urls import path
from . import views

urlpatterns=[
    path('old_image/',views.ImageList.as_view()),
    # path('old_image/<int:pk>',views.ImageDetail.as_view()),
    # path('new_image/',views.NImageList.as_view()),
    # path('new_image/<int:pk>',views.NImageDetail.as_view()),

]