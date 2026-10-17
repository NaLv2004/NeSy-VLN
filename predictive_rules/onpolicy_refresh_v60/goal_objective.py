"""Task-supervised endpoint equivalence, with entirely learned code IDs."""
import torch

def terminal_agreement(model,context,terminal_logits,alternate_features,goal_logits):
    alternate=model.observe(alternate_features[:,None],context)[:,0]
    own=terminal_logits.log_softmax(-1);other=alternate.log_softmax(-1)
    loss=-torch.logsumexp(own+other,dim=-1).mean()
    return loss,dict(terminal_pair_loss=loss.detach(),
        terminal_pair_agreement=(own.argmax(-1)==other.argmax(-1)).float().mean(),
        alternate_terminal_goal_agreement=(goal_logits.argmax(-1)==other.argmax(-1)).float().mean())
