"""Training-only sequential-versus-marginal classification of hard codes.

Adapted from the density-ratio objective of Allen et al., arXiv:2106.04379.
Unlike the original continuous fully-observed setting, the available source
is a pair of anonymous hard memory/observation IDs. No theorem is claimed.
"""
import torch
from torch.nn import functional as F
from model import hard

def temporal_ratio(model,memory,current,actual,actual_features,marginal_features,context):
    n=len(current);assert n>1
    # Half negatives share the recorded episode, avoiding a pure room-ID
    # discriminator; half match the original shuffled marginal estimator.
    shift=torch.randint(1,n,(),device=context.device)
    order=(torch.arange(n,device=context.device)+shift)%n
    across=actual_features[order]
    use_same=(torch.arange(n,device=context.device)%2==0)
    negative_features=torch.where(use_same[:,None,None,None],marginal_features,across)
    # IDs from another program cannot be copied directly: re-encode raw
    # feature evidence using this example's fixed definitions.
    negative=hard(model.observe(negative_features,context))
    real_input=torch.cat([memory,current,actual],-1)
    marginal_input=torch.cat([memory,current,negative],-1)
    real=model.temporal_contrast(real_input).squeeze(-1)
    marginal=model.temporal_contrast(marginal_input).squeeze(-1)
    loss=.5*(F.binary_cross_entropy_with_logits(real,torch.zeros_like(real))+
        F.binary_cross_entropy_with_logits(marginal,torch.ones_like(marginal)))
    exact=(negative_features.float()-actual_features.float()).square().mean((-1,-2))<1e-10
    metrics=dict(temporal_ratio_loss=loss.detach(),
        temporal_ratio_accuracy=.5*((real<0).float().mean()+(marginal>0).float().mean()),
        temporal_real_recall=(real<0).float().mean(),temporal_marginal_recall=(marginal>0).float().mean(),
        temporal_positive_negative_code_alias=(actual.argmax(-1)==negative.argmax(-1)).float().mean(),
        temporal_identical_negative_features=exact.float().mean(),temporal_same_episode_fraction=use_same.float().mean())
    return loss,metrics
