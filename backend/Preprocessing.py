import re

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch

MAX_LEN = 300  # Default sequence length


def remove_comments(code):
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code

def normalize_whitespace(code):
    code = code.replace('\t', ' ')
    code = code.replace('\n', ' ')
    code = " ".join(code.split())
    return code

def fix_encoding(code):
    code = code.encode('utf-8', errors='ignore').decode('utf-8')
    return code

def replace_literals(code):
    code = re.sub(r'".*?"', 'STRING_LITERAL', code)
    code = re.sub(r"'.*?'", 'CHAR_LITERAL', code)
    code = re.sub(r'\b\d+(\.\d+)?\b', 'NUMBER', code)
    return code

def preprocess_code(code):
    code = remove_comments(code)
    code = normalize_whitespace(code)
    code = fix_encoding(code)
    code = replace_literals(code)
    return code

def tokenize_code_regex(code):
    pattern = r'\w+|==|!=|<=|>=|[-+*/=<>%&|^~!;:(),{}[\]]'
    tokens = re.findall(pattern, code)
    return tokens

def encode_tokens(tokens, vocab):
    return [vocab.get(tok, vocab['<UNK>']) for tok in tokens]


# def prepare_data(df_train, df_validation):
#     # process df_train
#     df_train['padded_ids'] = pad_sequences(df_train['token_ids'], maxlen=MAX_LEN, padding='post', truncating='post').tolist()
#     X_train = torch.tensor(df_train['padded_ids'].tolist(), dtype=torch.long)
#     y_train = torch.tensor(df_train['target'].tolist(), dtype=torch.float)
    
#     # process df_validation
#     df_validation['padded_ids'] = pad_sequences(df_validation['token_ids'], maxlen=MAX_LEN, padding='post', truncating='post').tolist()
#     X_val = torch.tensor(df_validation['padded_ids'].tolist(), dtype=torch.long)
#     y_val = torch.tensor(df_validation['target'].tolist(), dtype=torch.float)
    
#     return X_train, y_train, X_val, y_val, MAX_LEN
