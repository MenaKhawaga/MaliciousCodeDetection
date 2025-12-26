# 🛡️ Malicious Code Detection for C/C++

A deep learning system to detect whether a C/C++ function is malicious or safe. Users can input a function via text or file upload and click **Analyze**; the system predicts the class along with a confidence score.

--- 

## 📑Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Dataset & DataLoader](#dataset--dataloader)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Deployment & Usage](#deployment--usage)
- [File Structure](#file-structure)
- [Team & Contributions](#team--contributions)
- [Acknowledgments](#acknowledgments)


---

## 🧩Project Overview

Our project is **Malicious Code Detection**. The user inputs a function written in C or C++, and by clicking **Analyze**, the system determines whether the code is malicious or safe, along with a confidence score.

To achieve this, we needed a model capable of understanding programming languages and token sequences in code. After research, we selected **CodeBERT**, a pretrained Transformer model trained on multiple programming languages such as Python, Java, PHP, and more.

The original model was neither trained for detecting malicious code nor specialized in C/C++. Therefore, we used **Transfer Learning**, leveraging the pretrained model's general knowledge of code, and applied **Fine-Tuning** on a custom C/C++ dataset labeled as:

- **Malicious** ⚠️  
- **Non-Malicious** ✅

The final objective is for the model to accurately classify whether a given function is **Malicious** or **Non-Malicious**.

---
## Data Collection 

- **Source:** Combined from multiple C/C++ datasets (Devign, LineVul, PrimeVul, Ours, MegaVul)  
- **Format:** Each row contains:  
  - `code` – the C/C++ function as a string  
  - `label` – 0 for Non-Malicious, 1 for Malicious  
- **Preprocessing:**  
  - Dropped unnecessary columns from original CSVs  
  - Standardized all datasets to have `code` and `label` columns  
  - Removed missing or empty functions  
- **Final Dataset Size:** 103,598 functions  

This ensures a clean and unified dataset ready for tokenization and model training.

---

## 🧹Data Preprocessing

- **Preprocessing:** Remove comments, normalize whitespace, fix encoding, replace literals, remove includes/macros  
- **Tokenization:** Convert code to token sequences using **CodeBERT tokenizer**  
- **Encoding:** Produce `input_ids` (numerical token IDs of the code) and `attention_mask` (indicates which tokens are real vs padding) for model input

---

## 📊Dataset & DataLoader

**Dataset:** Custom PyTorch `TensorDataset` combining:

- `input_ids` – numerical token IDs representing the code  
- `attention_mask` – indicates which tokens are real vs padding  
- `labels` – 0 for Non-Malicious ✅, 1 for Malicious ⚠️  

**Train/Validation/Test Split:**

- **Ratio:** 80% / 10% / 10%  
- **Samples:**  
  - Train: 82,877  
  - Validation: 10,360  
  - Test: 10,360  

**DataLoader:**

- **Batch size:** 16  
- **Shuffle:** Enabled during training 🔄  
- **Output:** Each batch returns `(input_ids, attention_mask, labels)` ready for model training and evaluation

---

## Model Architecture 

- **Base Model:**  CodeBERT (`microsoft/codebert-base`)  
- **Pooling:** Combines token embeddings into a single vector representing the whole function. CodeBERT outputs a vector for each token; pooling averages them to get one vector per function.  
- **Normalization:** Layer normalization applied to stabilize training ⚖️  
- **Classifier:**  
  - Linear layer (768 → 256)  
  - ReLU activation  
  - Dropout for regularization  
  - Final linear layer (256 → 1) to predict malicious or safe  
- **Output:** A single number (logit) per function, converted to probability  
- **Loss Function:** Binary cross-entropy with logits (`BCEWithLogitsLoss`)  
- **Optimizer:** AdamW ⚡

---

## 📈Training & Evaluation

- **Epochs:** 6  
- **Batch size:** 16  

**Training Steps:**

1. Input code is tokenized into `input_ids` and `attention_mask`  
2. Model predicts a logit for each function  
3. Apply loss function (`BCEWithLogitsLoss`) and update weights  
4. Convert logits to probability using sigmoid  
5. Probabilities ≥ 0.5 → Malicious ⚠️,Probabilities < 0.5 → Non-Malicious ✅  

**Evaluation:**  

- Metrics: Accuracy, Precision, Recall, F1-score  
- Calculated on training, validation, and test sets  

**Final Metrics:**

| Dataset      | Accuracy | Precision | Recall  | F1-Score |
|-------------|----------|-----------|---------|----------|
| Train       | 0.9304   | 0.8575    | 0.9780  | 0.9138   |
| Validation  | 0.7920   | 0.6843    | 0.8085  | 0.7413   |
| Test        | 0.7955   | 0.6917    | 0.8185  | 0.7498   |

---

## 🚀Deployment & Usage

### Running the System
1. Open the project folder in **Visual Studio Code (VS Code)**.

2. Open a terminal inside VS Code (`View > Terminal` or press `` Ctrl+` ``).

3. Make sure you are in the **backend** folder:
```bash
 cd backend
```
### Run the backend server
```bash
uvicorn main:app --port 9000
```
**--port 9000**: Runs the server on port 9000. You can change this if another service is using the port.

### The backend will now be available at: **http://localhost:9000**

### User Flow

1. Open index.html in a browser

2. Input a C/C++ function or upload a file

3. Click Analyze

4. The system outputs the prediction and confidence score
---

## 📁File Structure

```bash
Malicious-Code-Detection/
├── backend/
│   ├── main.py          # Backend server & inference
│   ├── Preprocessing.py    # Code preprocessing and tokenization
│   ├── model.py         # Model structure
│   └── model_weights.pt # Fine-tuned CodeBERT weights
├── frontend/
│   ├── index.html       # Web interface
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── Malicious_Code_Detection_Full_Pipeline.ipynb  # Full pipeline: data → preprocessing → model → train → save weights
└── README.md
```
---
## Team & Contributions

| Member Name     | Role / Contribution |
|-----------------|------------------|
| Mena Khawaga    | Backend: integrated model with backend and frontend |
| Aliaa Walid     | Frontend (HTML/CSS/JS) |
| Zaid Mohamed    | Data Collection, Splitting & Preprocessing   |
| Omar Adel       | Tokenization, DataLoader, `compute_metrics`(acc/precision/recall/F1) |
| Malak Abdelsattar| Model Architecture |
| Kenzy Elabd       | Training & Evaluation |

## 🔗Acknowledgments

- *CodeBERT model:* [Hugging Face – microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)  
- *CodeBERT paper:* [CodeBERT: A Pre-Trained Model for Programming and Natural Languages (arXiv:2002.08155)](https://arxiv.org/abs/2002.08155)
