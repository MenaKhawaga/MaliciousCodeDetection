import torch
import torch.nn as nn
from transformers import RobertaModel

class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.norm = nn.LayerNorm(768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def mean_pooling(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.norm(pooled)
        return self.classifier(pooled).squeeze(-1)
