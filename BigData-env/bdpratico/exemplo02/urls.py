from django.contrib import admin
from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('', views.index, name='index'),
    path('ia_import', views.ia_import, name='ia_import'),
    path('ia_import_save', views.ia_import_save, name='ia_import_save'),
    path('ia_import_list', views.ia_import_list, name='ia_import_list'),
    path('ia_knn_treino', views.ia_knn_treino, name='ia_knn_treino'), # Adiciona a rota para o treinamento
    path('ia_knn_matriz', views.ia_knn_matriz, name='ia_knn_matriz'), # Adiciona a rota para a matriz de confusão
    path('ia_knn_roc', views.ia_knn_roc, name='ia_knn_roc'), # Adiciona a rota para a curva ROC
    path('ia_knn_recall', views.ia_knn_recall, name='ia_knn_recall'), # Adiciona a rota para o recall
]

