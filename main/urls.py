from django.urls import path
# from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.index, name='main'),
    path('about_me', views.about, name='about'),
    path("simple_function", views.simple_function),
    path("python", views.button, name='python'),
    path("output", views.output, name='script'),
]