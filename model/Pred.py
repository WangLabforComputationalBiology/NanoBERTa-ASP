import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification
import random
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def setseed(seed):
    """Ensure unique results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(seed)


def nano_pred(chainseq, seed: int = 42):
    setseed(seed)
    tokenizer = RobertaTokenizer.from_pretrained('antibody-tokenizer', max_len=150)
    model = RobertaForTokenClassification.from_pretrained("nanoberta-ft")
    inputs = tokenizer(''.join(chainseq), return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    ls = outputs[0][0][1:-1]
    P = torch.softmax(ls, dim=1)
    v = P[:, 1]
    ps = v.detach().numpy()
    return ps
