from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.http import JsonResponse, HttpResponseBadRequest
from PIL import Image
import io
import re
from easyocr import Reader
import torch
from torchvision import transforms
from meme_analyzer.ml.model import MemeClassifierNet
from meme_analyzer.ml.nlp_correction import entity_preserving_correction

# Initialize OCR reader and model
reader = Reader(['en'], gpu=True)
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

def clean_text(text):
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_hateful_text(text: str) -> bool:
    return "muslim" in text.lower()

@method_decorator(csrf_exempt, name='dispatch')
class MemeSageView(View):
    def post(self, request):
        if 'image' not in request.FILES:
            return HttpResponseBadRequest("No image file found in request.")

        image = request.FILES['image']
        img = Image.open(image).convert('RGB')

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        ocr_results = reader.readtext(img_bytes)
        extracted_text = " ".join([text for (_, text, _) in ocr_results])
        cleaned_text = clean_text(extracted_text)

        corrected_text = entity_preserving_correction(cleaned_text) if cleaned_text else ""

        # Override label to hateful if keyword detected in corrected text
        if is_hateful_text(corrected_text):
            predicted_label = "Hateful"
        else:
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                label_idx = output.argmax(dim=1).item()
                predicted_label = label_map.get(label_idx, "Unknown")

        return JsonResponse({
            "extracted_text": corrected_text,
            "predicted_label": predicted_label,
        })

def home(request):
    from django.shortcuts import render
    return render(request, 'meme_analyzer/index.html')
