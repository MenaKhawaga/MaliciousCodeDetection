import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizerFast
from model import CodeBERTClassifier
from Preprocessing import preprocess_code
from typing import Optional
import time



app = FastAPI(title="Malicious Code Detection API")

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare the model
model = CodeBERTClassifier()    # instantiate model
model.to(device)                # move it the device(gpu or cpu)

# Load the saved model weights
state_dict_path = "model_weights.pt"  
state_dict = torch.load(state_dict_path, map_location=device)

# fix the model saved with DataParallel (keys have 'module.' prefix)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v


# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
MAX_LEN = 512

# load the cleaned state_dict
model.load_state_dict(new_state_dict)
model.eval()  # set to evaluation mode


# API Routes
@app.get("/")
def home():
    return {"message": "Welcome to Malicious Code Detection API"}

@app.get("/status")
def status():
    return {"status": "Backend is running"}

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

    # Preprocess the code input

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

    # Model Inference
    with torch.no_grad():
        logit = model(input_ids, attention_mask)
        probability = torch.sigmoid(logit).item()

    verdict = "malicious" if probability >= 0.5 else "safe"

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "verdict": verdict,            # malicious / safe
        "risk": round(probability, 4), # probability score
        "timeMs": elapsed_ms,
        "model": "CodeBERT fine-tuned"
    }
