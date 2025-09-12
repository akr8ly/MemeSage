from django.views import View
from django.http import JsonResponse, HttpResponseBadRequest
import easyocr
from PIL import Image
import io


reader = easyocr.Reader(['en'])

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

@method_decorator(csrf_exempt, name='dispatch')

class MemeSageView(View):
    def get(self, request):
        return JsonResponse({"message": "Welcome to MemeSage. Use POST to upload memes."})

    def post(self, request):
        if 'image' not in request.FILES:
            return HttpResponseBadRequest("No image file found in request.")

        image = request.FILES['image']
        img = Image.open(image)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        ocr_results = reader.readtext(img_bytes)
        extracted_text = " ".join([text for (_, text, _) in ocr_results])

        tags = ["non-hateful", "neutral"]
        confidence_scores = [0.85, 0.9]

        return JsonResponse({
            "extracted_text": extracted_text,
            "tags": tags,
            "confidence_scores": confidence_scores,
        })
