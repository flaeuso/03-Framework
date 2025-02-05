from django.shortcuts import render
from django.http import HttpResponse

from django.contrib.auth import authenticate, login
def index(request):    
    usuario = request.POST.get('username')
    senha = request.POST.get('password')
    user = authenticate(username=usuario, password=senha)
    if (user is not None):
        login(request, user)
        request.session['username'] = usuario
        request.session['password'] = senha
        request.session['usernamefull'] = user.get_full_name()
        print(request.session['username'])
        print(request.session['password'])
        print(request.session['usernamefull'])
        from django.shortcuts import redirect
        return redirect('pessoa_menu_alias')
    else:
        data = {}
        if (usuario):
            data['msg'] = "Usuário ou Senha Incorretos " + usuario
        return render(request, 'index.html', data)

from django_tables2 import SingleTableView
class pessoa_menu(SingleTableView):
    print("Aqui")
    from .models import pessoa
    from .tables import pessoa_table
    model = pessoa
    table_class = pessoa_table
    template_name_suffix = '_menu'
    table_pagination = {"per_page": 5}
    template_name = 'exemplo01/pessoa_menu.html'

def pagina0(request):
    return render(request, 'pagina0.html')

def pagina1(request):
    return render(request, 'pagina1.html')

def pagina2(request):
    from .models import pessoa
    dicionario = {}
    registros = pessoa.objects.all()
    dicionario['pessoas'] = registros
    return render(request, 'pagina2.html', dicionario)

def pagina3(request):
    from .models import pessoa
    dicionario = {}
    registros = pessoa.objects.all()
    dicionario['pessoas'] = registros
    return render(request, 'pagina3.html', dicionario)

def pagina4(request):
    nome = request.POST.get('nome')
    email = request.POST.get('email')
    celular = request.POST.get('celular')
    funcao = request.POST.get('funcao')
    nascimento = request.POST.get('nascimento')
    ativo = request.POST.get('ativo')
    print("Nome:", nome)
    print("eMail:", email)
    print("Celular:", celular)
    print("Funcao:", funcao)
    print("Nascimento:", nascimento)
    print("ativo:", ativo)
    return render(request, 'pagina4.html')

from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
class pessoa_create(CreateView):
    from .models import pessoa
    model = pessoa
    fields = ['nome', 'email', 'celular', 'funcao', 'nascimento', 'ativo']
    def get_success_url(self):
        return reverse_lazy('pessoa_menu_alias')
    
from django.views.generic import ListView
class pessoa_list(ListView):
    def dispatch(self, request, *args, **kwargs):
        if request.user.has_perm("exemplo01.view_pessoa"):
            return super().dispatch(request, *args, **kwargs)
        else:
            return HttpResponse("Sem permissão para listar pessoas")   
    from .models import pessoa
    model = pessoa
    queryset = pessoa.objects.filter(ativo=True)
    
from django.views.generic.edit import UpdateView
class pessoa_update(UpdateView):
    from .models import pessoa
    model = pessoa
    fields = ['nome', 'email', 'celular', 'funcao', 'nascimento', 'ativo']
    def get_success_url(self):
        return reverse_lazy('pessoa_menu_alias')
    
from django.views.generic.edit import DeleteView
class pessoa_delete(DeleteView):
    from .models import pessoa
    model = pessoa
    fields = ['nome', 'email', 'celular', 'funcao', 'nascimento', 'ativo']
    template_name_suffix = '_delete'
    def get_success_url(self):
        return reverse_lazy('pessoa_menu_alias')

def pagina5(request):
    if not(request.user.has_perm('exemplo01.add_pessoa')):
        return HttpResponse("Sem permissão para adicionar pessoas")    
    xnome = request.POST.get('nome')
    xemail = request.POST.get('email')
    xcelular = request.POST.get('celular')
    xfuncao = request.POST.get('funcao')
    xnascimento = request.POST.get('nascimento')
    xativo = request.POST.get('ativo')
    print("Nome:", xnome)
    print("eMail:", xemail)
    print("Celular:", xcelular)
    print("Funcao:", xfuncao)
    print("Nascimento:", xnascimento)
    print("ativo:", xativo)
    if (xnome is not None):
        if (xativo == 'on'):
            xativo = True
        else:
            xativo = False
        from .models import pessoa
        pessoa.objects.create(nome=xnome, email=xemail, celular=xcelular,
                          funcao=xfuncao, nascimento=xnascimento, ativo=xativo)
    return render(request, 'pagina5.html')



def pagina6(request):
    from .models import pessoa
    dicionario = {}
    registros = pessoa.objects.all()
    dicionario['pessoas'] = registros
    return render(request, 'pagina6.html', dicionario)

def pagina7(request):    
    dicionario = {}
    from datetime import datetime
    data_hora_atual = datetime.now()
    dicionario['data'] = data_hora_atual
    from .models import pessoa
    registros = pessoa.objects.all()
    dicionario['pessoas'] = registros    
    return render(request, 'pagina7.html', dicionario)

def pagina8(request):
    from .models import pessoa
    dicionario = {}
    registros = pessoa.objects.all()
    dicionario['pessoas'] = registros
    return render(request, 'exemplo01/listar_pessoas.html', dicionario)

def pagina9(request):
    import pandas as pd
    from .models import pessoa
    eixo_y = []
    p = pessoa.objects.all()
    for _regs in p:
        eixo_x = []
        eixo_x.append(_regs.id)
        eixo_x.append(_regs.nome)
        eixo_x.append(_regs.email)
        eixo_x.append(_regs.celular)
        eixo_x.append(_regs.nascimento)
        eixo_x.append(_regs.ativo)
        eixo_y.append(eixo_x)
    _rotulos_colunas = []
    _rotulos_colunas.append("id")
    _rotulos_colunas.append("nome")
    _rotulos_colunas.append("email")
    _rotulos_colunas.append("celular")
    _rotulos_colunas.append("nascimento")
    _rotulos_colunas.append("ativo")
    df = pd.DataFrame(eixo_y, columns=_rotulos_colunas)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=pessoas.csv'
    df.to_csv(path_or_buf=response)
    return response

def pagina10(request):
    import pandas as pd
    df=pd.read_csv('C:/Users/flaeu/Downloads/pessoas.csv',sep=',')
    for linha, coluna in df.iterrows():
        print(linha, "ID:", coluna['id'])
        print(linha, "Nome:", coluna['nome'])
        print(linha, "eMail:", coluna['email'])
        print(linha, "Celular:", coluna['celular'])
        print(linha, "Nascimento:", coluna['nascimento'])
        print(linha, "Ativo:", coluna['ativo'])
    return HttpResponse("Arquivo Importado")

def pagina11(request):
    from .models import exame
    import os
    from django.core.files.storage import FileSystemStorage
    if request.method == 'POST' and request.FILES['arq_upload']:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)
        print("upload", upload)
        print("file1", file1)
        print("file_url", file_url)
        file2 = open(file1,'r')
        for row in file2:
            colunas = row.replace("(", "").replace(")", "").split(",")
            exame.objects.create(valor=float(colunas[8]))
        file2.close()
        os.remove(file_url.replace("/", ""))
        return HttpResponse("Arquivo Importado")
    return render(request, 'pagina11.html')

def pagina12(request):
    from .models import exame
    import plotly.graph_objs as go
    from plotly.offline import plot
    exame_tmp = exame.objects.all()
    eixo_x = []
    eixo_y = []
    i=0
    for e in exame_tmp:
        i += 1
        eixo_x.append(i)
        eixo_y.append(e.valor)
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=eixo_x, y=eixo_y, mode='lines',
                                line_color='rgb(0, 0, 255)'))
    figura.update_layout(title="Dados de Exame", title_x=0.5, 
                         xaxis_title='Tempo', yaxis_title='Batimento Cardíaco')
    plot_div = plot(figura, output_type='div')
    dicionario = {}
    dicionario['grafico'] = plot_div
    return render(request, 'pagina12.html', dicionario)



