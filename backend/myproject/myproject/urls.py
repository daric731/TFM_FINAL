"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from analysis import views  # Import the views from the 'analysis' app


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Route for the root URL
    path('analysis/', include('analysis.urls')),  # Include URLs from the 'analysis' app
    path('data-analysis/', views.data_analysis, name='data_analysis'),
    path('predictions/', views.predictions_view, name='predictions'),  # Route for predictions view
]
