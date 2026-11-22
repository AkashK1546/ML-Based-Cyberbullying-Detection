"""Optional: download every model into ./models once (needs internet), then the app can run offline.

    python download_models.py
"""
import ml

ml.hub._load_text()
ml.hub._load_clip()
ml.hub._load_ocr()
print("\nAll models saved in:", ml.MODELS_DIR)
