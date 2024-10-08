from django.urls import path
from . import views

urlpatterns = [
    path('test', views.index, name='index'),
    path('', views.music_dashboard, name='music_dashboard'),
    path('discover/', views.discover, name='discover'),
    path('tunes/', views.tunes, name='tunes'),
    path('get-musical-features/', views.get_musical_features_data, name='get_musical_features_data'),
]
