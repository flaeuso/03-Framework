from django.contrib import admin
from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('', views.index, name='index'),
    path('ia_import', views.ia_import, name='ia_import'),
    path('ia_import_save', views.ia_import_save, name='ia_import_save'),
    path('ia_import_list', views.ia_import_list, name='ia_import_list'),
]

