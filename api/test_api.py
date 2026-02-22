"""
Test script for Cats-vs-Dogs Classification API
Run this after starting the Docker container. The script will upload a small
in-memory image to exercise the /predict endpoint.
"""
import requests
from io import BytesIO
from PIL import Image

# base url
BASE_URL = "http://localhost:8000"


def create_dummy_image(color=(255, 0, 0), size=(224, 224)):
    img = Image.new('RGB', size, color)
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


def test_health():
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    resp = requests.get(f"{BASE_URL}/health")
    print(resp.status_code, resp.json())
    assert resp.status_code == 200
    assert resp.json().get('status') == 'healthy'
    print(" Health check passed")


def test_model_info():
    print("\n" + "="*70)
    print("TEST 2: Model Info")
    print("="*70)
    resp = requests.get(f"{BASE_URL}/model/info")
    print(resp.status_code, resp.json())
    assert resp.status_code == 200
    print(" Model info retrieved")


def test_prediction():
    print("\n" + "="*70)
    print("TEST 3: Single Image Prediction")
    print("="*70)
    img_buf = create_dummy_image()
    files = {'file': ('red.jpg', img_buf, 'image/jpeg')}
    resp = requests.post(f"{BASE_URL}/predict", files=files)
    print(resp.status_code, resp.json())
    assert resp.status_code == 200
    data = resp.json()
    assert 'prediction' in data
    assert 'confidence' in data
    print(" Prediction endpoint working")


def run_all():
    test_health()
    test_model_info()
    test_prediction()


if __name__ == '__main__':
    run_all()
