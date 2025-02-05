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