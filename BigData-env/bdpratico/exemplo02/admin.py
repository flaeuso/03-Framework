from django.contrib import admin
from .models import Dados  # Importa o modelo Dados

@admin.register(Dados)
class DadosAdmin(admin.ModelAdmin):
    list_display = ('grupo', 'mdw', 'latw', 'tmcw', 'racw')  # Exibe esses campos na listagem
    search_fields = ('grupo',)  # Adiciona uma barra de pesquisa por grupo
    list_filter = ('grupo',)  # Adiciona filtros laterais
