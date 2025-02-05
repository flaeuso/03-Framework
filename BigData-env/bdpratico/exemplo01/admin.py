from django.contrib import admin

from .models import *

@admin.action(description="Habilitar Registros Selecionados")
def habilitar_pessoas(ModelAdmin, request, queryset):
    for p in queryset:
        p.ativo = True
        p.save()

@admin.action(description="Desabilitar Registros Selecionados") 
def desabilitar_pessoas(ModelAdmin, request, queryset):
    queryset.update(ativo=False)

class PessoaCustomizado(admin.ModelAdmin):
    list_display = ('nome', 'email', 'celular', 'funcao', 'nascimento', 'calcula_idade', 'ativo', )
    actions = [habilitar_pessoas, desabilitar_pessoas]
    
    @admin.display(description='Idade')
    def calcula_idade(self, obj):
        from datetime import date
        hoje = date.today()
        idade = hoje.year - obj.nascimento.year
        return idade

admin.site.register(pessoa, PessoaCustomizado)
admin.site.register(procedimento)
admin.site.register(procedimento_executado)