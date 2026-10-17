"""Program synthesis conditioned on automatically enumerated task tokens.

The vocabulary contains observed training instructions, not predicate meanings.
No visual tensor enters this module; its outputs remain executable programs.
"""
import torch
from torch import nn
from .small_generator import SymbolicProgramGenerator


class TaskTokenProgramGenerator(SymbolicProgramGenerator):
    def __init__(self,cfg,hidden=64,task_vocabulary=(),slots=4):
        super().__init__(cfg,hidden)
        tasks=sorted(set(task_vocabulary))
        if not tasks or any(not isinstance(t,str) or not t for t in tasks):raise ValueError('Training instruction vocabulary required')
        self.task_vocabulary=tasks;self.task_to_id={task:i for i,task in enumerate(tasks)}
        self.slots=slots;self.hidden=hidden
        self.instruction_tokens=nn.Embedding(len(tasks),slots*hidden)
        nn.init.normal_(self.instruction_tokens.weight,std=.2)
        self.instruction_norm=nn.LayerNorm(hidden)
        self.text.requires_grad_(False);self.text_encoder.requires_grad_(False)

    @classmethod
    def initialize_from(cls,original,cfg,task_vocabulary,hidden=64):
        model=cls(cfg,hidden,task_vocabulary).to(original.text.weight.device)
        result=model.load_state_dict(original.state_dict(),strict=False)
        assert set(result.missing_keys)=={'instruction_tokens.weight','instruction_norm.weight','instruction_norm.bias'}
        assert not result.unexpected_keys
        return model

    def context(self,task,images,feedback,previous=None):
        if task not in self.task_to_id:raise ValueError('Instruction absent from the training task vocabulary')
        index=torch.tensor([self.task_to_id[task]],device=self.instruction_tokens.weight.device)
        tokens=self.instruction_norm(self.instruction_tokens(index).reshape(1,self.slots,self.hidden))
        events=self._events(feedback)
        if events is None:return tokens
        history,_=self.history(self.events(events),tokens.mean(1)[None])
        return torch.cat([tokens,history],1)
