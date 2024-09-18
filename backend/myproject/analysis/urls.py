from django.urls import path
from . import views
from .views import calendar_view
urlpatterns = [
    path('', views.analysis_view, name='analysis'),
    path('test/', views.test_view, name='test_view'),
    path('data-analysis/', views.data_analysis, name='data_analysis'),
    path('test/', views.test_template_view, name='test_template_view'),
    path('test-template/', views.test_template_view, name='test_template'),
    path('calendar/', views.calendar_view, name='calendar'),
    path('predictions/', views.predictions_view, name='predictions'),
]
