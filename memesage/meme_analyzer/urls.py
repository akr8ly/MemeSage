from django.urls import path
from .views import MemeSageView

urlpatterns = [
    path('', MemeSageView.as_view(), name='memesage-root'),
]
