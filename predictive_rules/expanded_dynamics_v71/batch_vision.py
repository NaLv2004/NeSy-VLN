"""Batched frozen perception. Same preprocessing, rare roundoff shifts allowed."""
import torch
from torch.nn import functional as F

class BatchImageEncoder:
    def __init__(self,frontend,gate,batch_size=24):
        self.frontend=frontend;self.gate=gate;self.batch_size=batch_size
    @torch.no_grad()
    def encode(self,images):
        self.gate.check(self.frontend.vision,())
        from nesy_vla.runtime import image_tensor
        rows=[];v=self.frontend.vocabulary
        for start in range(0,len(images),self.batch_size):
            x=torch.cat([image_tensor(im,self.frontend.cfg['image_size'],'cuda') for im in images[start:start+self.batch_size]])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=F.adaptive_avg_pool2d(self.frontend.vision(x),(7,7)).flatten(2).transpose(1,2)
            rows.append(F.normalize(((f.float()-v['mean'])/v['sd'])@v['axes'],dim=-1).half())
        return torch.cat(rows)
