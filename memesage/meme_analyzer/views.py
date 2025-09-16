from meme_analyzer.ml.model import MemeClassifierNet
import torch
from torchvision import transforms
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View
from django.http import HttpResponseBadRequest  
from PIL import Image
import io
from easyocr import Reader
from django.shortcuts import render

import os
reader = Reader(['en'], gpu=False)  


# Load model once (global scope)
NUM_CLASSES = 3
MODEL_PATH = 'meme_analyzer/ml/classifier.pt'
model = MemeClassifierNet(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

label_map = {
    0: "Non-hateful",
    1: "Hateful"
}

@method_decorator(csrf_exempt, name='dispatch')
class MemeSageView(View):
    def get(self, request):
        return JsonResponse({"message": "Welcome to MemeSage. Use POST to upload memes."})

    def post(self, request):
        if 'image' not in request.FILES:
            return HttpResponseBadRequest("No image file found in request.")

        image = request.FILES['image']
        img = Image.open(image).convert('RGB')

        # OCR Extraction (keep your existing)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        ocr_results = reader.readtext(img_bytes)
        extracted_text = " ".join([text for (_, text, _) in ocr_results])

        # Model prediction
        img_transformed = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_transformed)
            pred_label = output.argmax(dim=1).item()
            pred_class = label_map.get(pred_label, "Unknown")

        # Return combined OCR text + predicted label
        return JsonResponse({
            "extracted_text": extracted_text,
            "predicted_label": pred_class,
        })

def home(request):
    return render(request, 'meme_analyzer/index.html')
