from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import torch
import time

from transformers import RobertaTokenizerFast
from model import CodeBERTClassifier
from Preprocessing import preprocess_code


# =====================================================
# FastAPI App
# =====================================================
app = FastAPI(title="Malicious Code Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize your model
model = CodeBERTClassifier()
model.to(device)

# load the saved state_dict
state_dict_path = "model_weights.pt"  

# load the state_dict safely
state_dict = torch.load(state_dict_path, map_location=device)

# fix if model was saved with DataParallel (keys have 'module.' prefix)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # remove 'module.' if it exists
    name = k.replace("module.", "")
    new_state_dict[name] = v


# Tokenizer
# ------------------------
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
MAX_LEN = 512

# load the cleaned state_dict
model.load_state_dict(new_state_dict)
model.eval()  # set to evaluation mode
# =====================================================
# Routes
# =====================================================
@app.get("/")
def home():
    return {"message": "Malicious Code Detection API is running"}


@app.get("/status")
def status():
    return {
        "status": "ok",
        "device": str(device),
        "model": "CodeBERT (fine-tuned)"
    }


# =====================================================
# Prediction Endpoint
# =====================================================
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    code: Optional[str] = Form(None)
):
    if file:
        code_bytes = await file.read()
        code = code_bytes.decode("utf-8", errors="ignore")
    elif code:
        code = code
    else:
        return {"error": "No code provided"}

    start_time = time.time()

    # -----------------------
    # Preprocessing
    code = preprocess_code(code)

    encoded = tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # -----------------------
    # Inference
    with torch.no_grad():
        logit = model(input_ids, attention_mask)
        probability = torch.sigmoid(logit).item()

    verdict = "malicious" if probability >= 0.5 else "safe"

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "verdict": verdict,          # malicious / safe
        "risk": round(probability, 4),
        "timeMs": elapsed_ms,
        "model": "CodeBERT fine-tuned"
    }
