from django.shortcuts import render

app_name = 'exemplo02'
from django.http import HttpResponse
def index(request):
    return HttpResponse("AGORA EH O EXEMPLO 02.")