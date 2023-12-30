from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('/classifier', views.classifier, name='classifier'),
    path('/generator', views.generator, name='generator'),
]
