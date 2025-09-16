from django.urls import path
from .views import home, MemeSageView

urlpatterns = [
    path('', home, name='home'),
    path('api/meme/', MemeSageView.as_view()),  
]

