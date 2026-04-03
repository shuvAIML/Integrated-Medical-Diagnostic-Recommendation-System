import requests

def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        # Using free OCR.space API for lightweight text extraction
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': ('report.jpg', image_bytes, 'image/jpeg')},
            data={'isOverlayRequired': False, 'apikey': 'helloworld', 'language': 'eng'},
            timeout=15
        )
        result = response.json()
        if result.get('ParsedResults'):
            return result['ParsedResults'][0]['ParsedText']
        return "No readable text found in document."
    except Exception as e:
        return f"Cloud OCR Error: {str(e)}"