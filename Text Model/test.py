import os
import re
import joblib
import torch
import emoji

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "distilbert_cyberbullying_finetuned"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.joblib"
)

# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking project files...")

if not os.path.exists(MODEL_DIR):
    print("\nERROR: Model folder not found:")
    print(MODEL_DIR)
    input("\nPress Enter to exit...")
    exit()

if not os.path.exists(LABEL_ENCODER_PATH):
    print("\nERROR: label_encoder.joblib not found:")
    print(LABEL_ENCODER_PATH)
    input("\nPress Enter to exit...")
    exit()

print("Model folder      : FOUND")
print("Label encoder     : FOUND")

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device             :", device)

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_DIR
)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("Loading label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print("\nModel loaded successfully!")

print("\nCyberbullying categories:")

for label in label_encoder.classes_:
    print(" -", label)

# ============================================================
# TEXT CLEANING
# Same cleaning approach used during training
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert emojis to text
    text = emoji.demojize(text)

    # Replace emoji formatting
    text = text.replace(":", " ")
    text = text.replace("_", " ")

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_text(text):

    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, predicted_id = torch.max(
            probabilities,
            dim=1
        )

    predicted_id = predicted_id.item()

    confidence = confidence.item()

    predicted_label = label_encoder.inverse_transform(
        [predicted_id]
    )[0]

    return predicted_label, confidence

# ============================================================
# INTERACTIVE TESTING
# ============================================================

print("\n" + "=" * 60)
print("       CYBERBULLYING DETECTION SYSTEM")
print("=" * 60)

print("\nEnter a tweet/text to classify.")
print("Type 'exit' to stop.")

while True:

    text = input("\nEnter text: ")

    if text.lower().strip() == "exit":
        print("\nExiting...")
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    try:

        label, confidence = predict_text(text)

        print("\n" + "-" * 60)
        print("Prediction")
        print("-" * 60)

        print("Text       :", text)
        print("Category   :", label)
        print(
            "Confidence : {:.2f}%".format(
                confidence * 100
            )
        )

        print("-" * 60)

    except Exception as e:

        print("\nPrediction error:")
        print(e)
