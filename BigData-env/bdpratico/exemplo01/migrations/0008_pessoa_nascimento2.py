# Generated by Django 5.1.4 on 2025-01-10 12:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('exemplo01', '0007_alter_pessoa_ativo'),
    ]

    operations = [
        migrations.AddField(
            model_name='pessoa',
            name='nascimento2',
            field=models.DateField(blank=True, null=True, verbose_name='Nascimento'),
        ),
    ]
