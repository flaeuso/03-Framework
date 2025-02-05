from django.db import models

class Dados(models.Model):
    grupo = models.CharField(
        max_length=50, null=True, blank=True, verbose_name='Grupo'
    )
    mdw = models.FloatField(null=True, blank=True, verbose_name='mdw')
    latw = models.FloatField(null=True, blank=True, verbose_name='latw')
    tmcw = models.FloatField(null=True, blank=True, verbose_name='tmcw')
    racw = models.FloatField(null=True, blank=True, verbose_name='racw')
    araw = models.FloatField(null=True, blank=True, verbose_name='araw')
    mcw = models.FloatField(null=True, blank=True, verbose_name='mcw')
    psdsw = models.FloatField(null=True, blank=True, verbose_name='psdsw')
    s6w = models.FloatField(null=True, blank=True, verbose_name='s6w')
    mdr = models.FloatField(null=True, blank=True, verbose_name='mdr')
    latr = models.FloatField(null=True, blank=True, verbose_name='latr')
    tmcr = models.FloatField(null=True, blank=True, verbose_name='tmcr')
    racr = models.FloatField(null=True, blank=True, verbose_name='racr')
    arar = models.FloatField(null=True, blank=True, verbose_name='arar')
    mcr = models.FloatField(null=True, blank=True, verbose_name='mcr')
    psdsr = models.FloatField(null=True, blank=True, verbose_name='psdsr')
    s6r = models.FloatField(null=True, blank=True, verbose_name='s6r')
    mdg = models.FloatField(null=True, blank=True, verbose_name='mdg')
    latg = models.FloatField(null=True, blank=True, verbose_name='latg')
    tmcg = models.FloatField(null=True, blank=True, verbose_name='tmcg')
    racg = models.FloatField(null=True, blank=True, verbose_name='racg')
    arag = models.FloatField(null=True, blank=True, verbose_name='arag')
    mcg = models.FloatField(null=True, blank=True, verbose_name='mcg')
    psdsg = models.FloatField(null=True, blank=True, verbose_name='psdsg')
    s6g = models.FloatField(null=True, blank=True, verbose_name='s6g')
    mdwb = models.FloatField(null=True, blank=True, verbose_name='mdwb')
    latb = models.FloatField(null=True, blank=True, verbose_name='latb')
    tmcb = models.FloatField(null=True, blank=True, verbose_name='tmcb')
    racb = models.FloatField(null=True, blank=True, verbose_name='racb')
    arab = models.FloatField(null=True, blank=True, verbose_name='arab')
    mcb = models.FloatField(null=True, blank=True, verbose_name='mcb')
    psdsb = models.FloatField(null=True, blank=True, verbose_name='psdsb')
    s6b = models.FloatField(null=True, blank=True, verbose_name='s6b')

    def __str__(self):
        return self.grupo

    class Meta:
        ordering = ['grupo']# Create your models here.
