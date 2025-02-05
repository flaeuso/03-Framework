from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('exemplo01.urls')),
    path('exemplo01/', include('exemplo01.urls')),
    path('exemplo02/', include('exemplo02.urls')),
    
]
