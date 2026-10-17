"""Reference 1.5-entmax and proper loss, isolated from live training imports."""
from pathlib import Path
import sys
import torch
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'vendor_entmax_1_3'))
from entmax import entmax15,entmax15_loss

def probability(scores):
    return entmax15(scores,dim=-1)

def log_probability(scores):
    probabilities=probability(scores)
    return torch.where(probabilities>0,probabilities.clamp_min(1e-30).log(),-torch.inf)

def loss(scores,labels,mask=None):
    value=entmax15_loss(scores.flatten(0,-2),labels.flatten()).reshape(labels.shape)
    return value.mean() if mask is None else (value*mask).sum()/mask.sum().clamp_min(1)
