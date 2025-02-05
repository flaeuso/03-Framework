from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import Dados  # Certifique-se de que o nome da classe do modelo está correto
import os

def index(request):
    return render(request, 'index.html')

def ia_import(request):
    return render(request, 'ia_import.html')


def ia_import_save(request):
    if request.method == 'POST' and request.FILES.get('arq_upload'):
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)

        # Apaga todos os registros antes de importar novos dados
        Dados.objects.all().delete()

        i = 0
        file_path = os.path.join(fss.location, file1)

        with open(file_path, 'r') as file2:
            for row in file2:
                if i > 0:
                    row2 = row.replace(',', '.')
                    row3 = row2.split(';')

                    Dados.objects.create(
                        grupo=row3[0],
                        mdw=float(row3[1]), latw=float(row3[2]),
                        tmcw=float(row3[3]), racw=float(row3[4]),
                        araw=float(row3[5]), mcw=float(row3[6]),
                        psdsw=float(row3[7]), s6w=float(row3[8]),
                        mdr=float(row3[9]), latr=float(row3[10]),
                        tmcr=float(row3[11]), racr=float(row3[12]),
                        arar=float(row3[13]), mcr=float(row3[14]),
                        psdsr=float(row3[15]), s6r=float(row3[16]),
                        mdg=float(row3[17]), latg=float(row3[18]),
                        tmcg=float(row3[19]), racg=float(row3[20]),
                        arag=float(row3[21]), mcg=float(row3[22]),
                        psdsg=float(row3[23]), s6g=float(row3[24]),
                        mdwb=float(row3[25]), latb=float(row3[26]),
                        tmcb=float(row3[27]), racb=float(row3[28]),
                        arab=float(row3[29]), mcb=float(row3[30]),
                        psdsb=float(row3[31]), s6b=float(row3[32])
                    )
                i += 1

        # Remove o arquivo temporário após a importação
        os.remove(file_path)

    return redirect('ia_import_list')  # Redireciona para a listagem após importar


def ia_import_list(request):
    data = {'dados': Dados.objects.all()}
    return render(request, 'ia_import_list.html', data)

def ia_knn_treino(request):
    # Dicionário para armazenar os dados que serão passados ao template
    data = {}
    print("Vamos ao que interessa...")

    # Importando bibliotecas necessárias
    import pandas as pd
    from .models import Dados  # Importando o modelo de dados do Django

    # Obtendo todos os registros do banco de dados
    dados_queryset = Dados.objects.all()
    print("Registros Selecionados.")

    # Convertendo os dados do banco para um DataFrame do Pandas
    df = pd.DataFrame(list(dados_queryset.values()))
    print("Pandas Carregado e dados 'convertidos'.")
    print("'Cabeçalho' dos dados:")
    print(df.head())

    # Importando biblioteca para dividir os dados em conjuntos de treino, teste e validação
    from sklearn.model_selection import train_test_split
    print("Sklearn carregado")

    # Definição das variáveis independentes (X) e variável alvo (y)
    X = df.drop(columns=['grupo', 'id'])  # Remove colunas 'grupo' (target) e 'id' (não é relevante para o modelo)
    y = df['grupo']  # Variável alvo

    # Divisão dos dados: 70% treino, 15% teste e 15% validação
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # Armazenando informações sobre o tamanho dos conjuntos
    data['dataset'] = X_train.shape
    data['treino'] = X_train.shape[0]
    data['teste'] = X_test.shape[0]
    data['validacao'] = X_val.shape[0]

    print(f'Tamanho do conjunto de treino: {X_train.shape}')
    print(f'Tamanho do conjunto de teste: {X_test.shape}')
    print(f'Tamanho do conjunto de validação: {X_val.shape}')

    # Importando bibliotecas para o modelo KNN
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    # Instanciando o modelo KNN
    knn = KNeighborsClassifier()

    # Definição dos parâmetros para otimização com GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],  # Número de vizinhos a considerar
        'weights': ['uniform', 'distance'],  # Tipo de ponderação dos vizinhos
        'metric': ['euclidean', 'manhattan']  # Métrica de distância usada no cálculo
    }

    # GridSearchCV para encontrar os melhores parâmetros
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    # Treinando o modelo com os dados de treino
    grid_search.fit(X_train, y_train)

    # Melhor conjunto de parâmetros encontrados pelo GridSearchCV
    data['best'] = grid_search.best_params_
    print("Melhores parâmetros encontrados:", grid_search.best_params_)

    # Obtendo o melhor modelo treinado
    best_knn = grid_search.best_estimator_

    # Fazendo previsões no conjunto de validação
    y_val_pred = best_knn.predict(X_val)

    # Avaliação da acurácia no conjunto de validação
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Acurácia no conjunto de validação: {val_accuracy * 100:.2f}%')
    data['acc_validacao'] = round(val_accuracy * 100, 2)

    # Fazendo previsões no conjunto de teste
    y_test_pred = best_knn.predict(X_test)

    # Avaliação da acurácia no conjunto de teste
    test_accuracy = accuracy_score(y_test, y_test_pred)
    data['acc_teste'] = round(test_accuracy * 100, 2)
    print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

    # Importando biblioteca para salvar o modelo treinado
    import joblib

    # Salvando o modelo treinado em um arquivo .pkl
    model_filename = 'knn_model.pkl'  # Nome do arquivo
    joblib.dump(best_knn, model_filename)
    print(f'Modelo salvo em: {model_filename}')
    data['file'] = model_filename

    # Renderiza a página HTML com os resultados
    return render(request, 'ia_knn_treino.html', data)

def ia_knn_matriz(request):
    # Importando bibliotecas necessárias
    import joblib
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    from .models import Dados  # Importando o modelo de dados do Django

    # Obtendo todos os registros do banco de dados
    dados_queryset = Dados.objects.all()
    
    # Convertendo os dados do banco para um DataFrame do Pandas
    df = pd.DataFrame(list(dados_queryset.values()))
    
    # Importando biblioteca para dividir os dados em treino e teste
    from sklearn.model_selection import train_test_split

    # Definição das variáveis independentes (X) e variável alvo (y)
    X = df.drop(columns=['grupo', 'id'])  # Remove colunas 'grupo' (target) e 'id' (não é relevante para o modelo)
    y = df['grupo']  # Variável alvo

    # Nome do arquivo onde o modelo KNN foi salvo
    model_filename = 'knn_model.pkl'
    
    # Carregando o modelo treinado
    best_knn = joblib.load(model_filename)

    # Fazendo previsões no conjunto completo de dados
    y_pred = best_knn.predict(X)

    # Criando a matriz de confusão
    cm = confusion_matrix(y, y_pred)
    
    # Estruturando os dados para passar ao template
    data = {
        'matrix': cm.tolist(),  # Convertendo a matriz para uma lista
        'labels': np.unique(y).tolist()  # Pegando os rótulos únicos
    }
    
    # Exibindo a matriz de confusão no console
    for i in data['matrix']:
        print(i)
    
    # Renderiza a página HTML com os resultados da matriz de confusão
    return render(request, 'ia_knn_matriz.html', data)

def ia_knn_roc(request):
    # Importando bibliotecas necessárias
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go
    import numpy as np
    from .models import Dados
    from django.shortcuts import render

    # Obtendo todos os registros do banco de dados
    dados_queryset = Dados.objects.all()
    
    # Convertendo os dados do banco para um DataFrame do Pandas
    df = pd.DataFrame(list(dados_queryset.values()))

    # Definição das variáveis independentes (X) e variável alvo (y)
    X = df.drop(columns=['grupo', 'id'])  # Remove colunas 'grupo' (target) e 'id' (não é relevante para o modelo)
    y = df['grupo'].map({'Controle': -1, 'Experimental': 1})  # Mapeando os valores para -1 e 1

    # Nome do arquivo onde o modelo KNN foi salvo
    model_filename = 'knn_model.pkl'
    
    # Carregando o modelo treinado
    best_knn = joblib.load(model_filename)

    # Obtendo as probabilidades previstas pelo modelo
    y_pred_prob = best_knn.predict_proba(X)[:, 1]

    # Calculando a curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Criando a figura da Curva ROC usando Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})',
        line=dict(color='blue')
    ))
    
    # Adicionando a linha do classificador aleatório
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    # Configurando o layout do gráfico
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos (FPR)',
        yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
        showlegend=True
    )
    
    # Convertendo o gráfico para HTML para renderização no template
    graph = fig.to_html(full_html=False)
    
    # Renderiza a página HTML com o gráfico da Curva ROC
    return render(request, 'ia_knn_roc.html', {'graph': graph})

def ia_knn_recall(request):
    # Importando bibliotecas necessárias
    import joblib
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, auc
    import plotly.graph_objects as go
    import numpy as np
    from .models import Dados
    from django.shortcuts import render

    # Obtendo todos os registros do banco de dados
    dados_queryset = Dados.objects.all()
    
    # Convertendo os dados do banco para um DataFrame do Pandas
    df = pd.DataFrame(list(dados_queryset.values()))

    # Definição das variáveis independentes (X) e variável alvo (y)
    X = df.drop(columns=['grupo', 'id'])  # Remove colunas 'grupo' (target) e 'id' (não é relevante para o modelo)
    y = df['grupo'].map({'Controle': -1, 'Experimental': 1})  # Mapeando os valores para -1 e 1

    # Nome do arquivo onde o modelo KNN foi salvo
    model_filename = 'knn_model.pkl'
    
    # Carregando o modelo treinado
    best_knn = joblib.load(model_filename)

    # Obtendo as probabilidades previstas pelo modelo
    y_pred_prob = best_knn.predict_proba(X)[:, 1]

    # Calculando a curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y, y_pred_prob)
    pr_auc = auc(recall, precision)

    # Criando a figura da Curva Precision-Recall usando Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines',
        name=f'Precision-Recall Curve (AUC = {pr_auc:.2f})',
        line=dict(color='blue')
    ))
    
    # Adicionando a linha do classificador aleatório
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    # Configurando o layout do gráfico
    fig.update_layout(
        title='Curva Precision-Recall',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    
    # Convertendo o gráfico para HTML para renderização no template
    graph = fig.to_html(full_html=False)
    
    # Renderiza a página HTML com o gráfico da Curva Precision-Recall
    return render(request, 'ia_knn_recall.html', {'graph': graph})
