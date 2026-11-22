
import os
import threading

import emoji_analysis

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Change with environment variables, e.g.  set TEXT_MODEL=martin-ha/toxic-comment-model  (lighter)
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL", "unitary/toxic-bert")
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
ENABLE_OCR = os.environ.get("ENABLE_OCR", "1") == "1"
LOAD_TIMEOUT = 900

# All pretrained models are stored INSIDE the project, in ./models/<model-name>/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def local_dir(name):
    """models/unitary__toxic-bert  etc."""
    return os.path.join(MODELS_DIR, name.replace("/", "__"))


def load_pretrained(name, *classes):
    """Load from ./models if already saved there, otherwise download once and save it there."""
    path = local_dir(name)
    if os.path.isdir(path) and os.listdir(path):
        print(f"📁 Loading {name} from {path}")
        return [c.from_pretrained(path) for c in classes]
    print(f"⬇️  Downloading {name} (first run only) -> {path}")
    objs = [c.from_pretrained(name) for c in classes]
    os.makedirs(path, exist_ok=True)
    for o in objs:
        o.save_pretrained(path)
    return objs          # first start downloads the models, which can take a while


class ModelsUnavailable(RuntimeError):
    pass


class ModelHub:
    def __init__(self):
        self.text_tok = self.text_model = None
        self.text_toxic_idx = []
        self.text_multilabel = False
        self.clip_model = self.clip_processor = None
        self.ocr = None
        self.errors = {}
        self.ready = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        self.infer_lock = threading.Lock()

    # ---- loading ------------------------------------------------------
    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._load_all, daemon=True, name="model-loader").start()

    def _load_all(self):
        for name, fn in (("text", self._load_text), ("clip", self._load_clip), ("ocr", self._load_ocr)):
            try:
                fn()
            except Exception as e:                        # keep going, report all problems together
                self.errors[name] = f"{type(e).__name__}: {e}"
                print(f"❌ Could not load {name} model: {self.errors[name]}")
        self.ready.set()
        print("🚀 Model loading finished" + ("" if not self.errors else f" - PROBLEMS: {list(self.errors)}"))

    def _load_text(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"🔄 Loading text model {TEXT_MODEL_NAME} ...")
        self.text_tok, self.text_model = load_pretrained(
            TEXT_MODEL_NAME, AutoTokenizer, AutoModelForSequenceClassification)
        self.text_model.eval()
        cfg = self.text_model.config
        id2label = {int(i): str(l).lower() for i, l in cfg.id2label.items()}
        safe = {"non-toxic", "non_toxic", "nontoxic", "neutral", "normal", "not_offensive",
                "non-offensive", "not_hate", "label_0", "clean", "benign", "none"}
        self.text_toxic_idx = [i for i, l in id2label.items() if l not in safe] or [len(id2label) - 1]
        self.text_multilabel = (getattr(cfg, "problem_type", None) == "multi_label_classification"
                                or "toxic-bert" in TEXT_MODEL_NAME or len(id2label) == 1)
        print(f"✅ Text model ready (toxic labels: {[id2label[i] for i in self.text_toxic_idx]})")

    def _load_clip(self):
        from transformers import CLIPProcessor, CLIPModel
        print(f"🔄 Loading CLIP {CLIP_MODEL_NAME} ...")
        self.clip_processor, self.clip_model = load_pretrained(CLIP_MODEL_NAME, CLIPProcessor, CLIPModel)
        self.clip_model.eval()
        print("✅ CLIP ready")

    def _load_ocr(self):
        if not ENABLE_OCR:
            print("ℹ️ OCR disabled (ENABLE_OCR=0): text inside images will not be analysed")
            return
        import easyocr
        ocr_dir = os.path.join(MODELS_DIR, "easyocr")
        os.makedirs(ocr_dir, exist_ok=True)
        self.ocr = easyocr.Reader(['en'], gpu=False, verbose=False,
                                  model_storage_directory=ocr_dir, user_network_directory=ocr_dir)
        print("✅ EasyOCR ready")

    # ---- readiness ----------------------------------------------------
    def ensure_ready(self):
        self.start()
        if not self.ready.wait(LOAD_TIMEOUT):
            raise ModelsUnavailable("The AI moderation models are still loading. Please try again in a minute.")
        needed = ["text", "clip"] + (["ocr"] if ENABLE_OCR else [])
        bad = [n for n in needed if n in self.errors]
        if bad:
            raise ModelsUnavailable(
                "AI moderation is unavailable, so posting and commenting are disabled. Failed to load: "
                + "; ".join(f"{n} ({self.errors[n]})" for n in bad))


hub = ModelHub()


def ensure_ready():
    hub.ensure_ready()


# ---------------------------------------------------------------------------
# Text toxicity / cyberbullying (pretrained transformer)
# ---------------------------------------------------------------------------
def toxicity_score(text):
    """Return a cyberbullying/toxicity probability in [0, 1]."""
    text = (text or "").strip()
    if not text:
        return 0.0
    hub.ensure_ready()
    import torch
    with hub.infer_lock, torch.inference_mode():
        enc = hub.text_tok(text, return_tensors="pt", truncation=True, max_length=256)
        logits = hub.text_model(**enc).logits[0]
        probs = torch.sigmoid(logits) if hub.text_multilabel else torch.softmax(logits, dim=-1)
    toxic = [float(probs[i]) for i in hub.text_toxic_idx]
    return max(toxic) if hub.text_multilabel else min(1.0, sum(toxic))


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def extract_text_from_image(image_path):
    """Return dict(text, confidence, word_count, detections). Empty text if no OCR / no text."""
    empty = {'text': '', 'confidence': 0.0, 'word_count': 0, 'detections': 0}
    if not ENABLE_OCR:
        return empty
    hub.ensure_ready()
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image for OCR")
    h, w = img.shape[:2]
    scale = 1280 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    with hub.infer_lock:
        results = hub.ocr.readtext(img)
    texts = [t for (_b, t, _c) in results]
    confs = [c for (_b, _t, c) in results]
    combined = " ".join(texts)
    return {'text': combined,
            'confidence': float(sum(confs) / len(confs)) if confs else 0.0,
            'word_count': len(combined.split()),
            'detections': len(results)}


# ---------------------------------------------------------------------------
# CLIP zero-shot image moderation
# ---------------------------------------------------------------------------
CLIP_CATEGORIES = {
    'cyberbullying': ["cyberbullying or targeted online harassment",
                      "not cyberbullying and not targeted harassment"],
    'sentiment': ["positive and friendly sentiment", "neutral sentiment",
                  "negative and hostile sentiment"],
    'sarcasm': ["sarcastic or ironic meaning", "literal and non-sarcastic meaning"],
    'harmfulness': ["completely harmless content",
                    "partially harmful or mildly harmful content",
                    "very harmful, threatening or severely abusive content"],
    'offensive': ["offensive, insulting or abusive content", "non-offensive content"],
    'violence': ["violent, threatening or physically aggressive content", "non-violent content"],
    'hate': ["hate speech or targeted abuse against a person or group",
             "no hate speech or targeted abuse"],
    'overall': ["cyberbullying or online harassment", "offensive content", "harmful content",
                "harmless normal content", "humorous meme or joke"],
    # Emoji analysis of the picture itself (labels live in emoji_analysis.py, which scores them)
    'emoji_presence': emoji_analysis.CLIP_PRESENCE_LABELS,
    'emoji_sentiment': emoji_analysis.CLIP_SENTIMENT_LABELS,
}
CLIP_WEIGHTS = {'cyberbullying': 2.5, 'offensive': 1.5, 'hate': 2.0, 'violence': 1.5}


def _clip_probs(image_path):
    import torch
    from PIL import Image
    hub.ensure_ready()
    image = Image.open(image_path).convert("RGB")
    flat, spans = [], {}
    for cat, labels in CLIP_CATEGORIES.items():
        spans[cat] = (len(flat), len(flat) + len(labels))
        flat += [f"a social media image showing {l}" for l in labels]
    with hub.infer_lock, torch.inference_mode():
        inputs = hub.clip_processor(text=flat, images=image, return_tensors="pt",
                                    padding=True, truncation=True)
        logits = hub.clip_model(**inputs).logits_per_image[0]
    out = {}
    for cat, (a, b) in spans.items():
        p = torch.softmax(logits[a:b], dim=0)
        out[cat] = {l: float(x) * 100 for l, x in zip(CLIP_CATEGORIES[cat], p)}
    return out


def _clip_score(clip):
    total = wsum = 0.0
    for cat, w in CLIP_WEIGHTS.items():
        total += list(clip[cat].values())[0] / 100.0 * w
        wsum += w
    harm = list(clip['harmfulness'].values())          # harmless / partial / very
    total += (harm[2] / 100.0 + harm[1] / 200.0) * 2.0
    wsum += 2.0
    total += list(clip['sentiment'].values())[2] / 100.0
    wsum += 1.0
    return min(1.0, total / wsum)


def predict_image_full(image_path):
    """Return (predictions, score, clip_probs).

    predictions = {category: {predicted, confidence}};  clip_probs = {category: {label: percent}}.
    """
    clip = _clip_probs(image_path)
    preds = {}
    for cat, probs in clip.items():
        top = max(probs, key=probs.get)
        preds[cat] = {'predicted': top, 'confidence': probs[top] / 100.0}
    return preds, _clip_score(clip), clip


def predict_image(image_path):
    """Return (predictions, score). predictions = {category: {predicted, confidence}}."""
    preds, score, _clip = predict_image_full(image_path)
    return preds, score


def emoji_in_image(clip_probs):
    """Emoji analysis of the picture, from the CLIP probabilities (see emoji_analysis.analyze_image_clip)."""
    return emoji_analysis.analyze_image_clip(clip_probs['emoji_presence'], clip_probs['emoji_sentiment'])
