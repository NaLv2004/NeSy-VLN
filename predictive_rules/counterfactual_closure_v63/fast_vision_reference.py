"""Single-frame CUDA-graph encoding, with an explicit boundary gate.

The graph preserves the original image layout and one-frame operations rather
than changing the convolution batch size. It never executes between revisions.
"""
import torch
from torch.nn import functional as F


class BoundaryGraphEncoder:
    def __init__(self,frontend,gate):
        self.frontend=frontend;self.gate=gate;self.graph=None;self.input=None;self.output=None
        self.calls=0;self.frames=0

    def _encode_tensor(self):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=F.adaptive_avg_pool2d(self.frontend.vision(self.input),(7,7)).flatten(2).transpose(1,2)
        vocabulary=self.frontend.vocabulary
        z=F.normalize(((features.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
        return torch.cdist(z.flatten(0,1),vocabulary['centers']).argmin(1)

    @torch.no_grad()
    def encode(self,images):
        # CUDA replay does not trigger module hooks. Require the same gate
        # explicitly before preprocessing an observation or replaying kernels.
        self.gate.check(self.frontend.vision,())
        from nesy_vla.runtime import image_tensor
        self.calls+=1;self.frames+=len(images);grids=[]
        for image in images:
            source=image_tensor(image,self.frontend.cfg['image_size'],'cuda')
            if self.graph is None:
                self.input=torch.empty_strided(source.shape,source.stride(),dtype=source.dtype,device=source.device)
                self.input.copy_(source)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):self.output=self._encode_tensor()
                torch.cuda.current_stream().wait_stream(stream)
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):self.output=self._encode_tensor()
            else:self.input.copy_(source)
            self.gate.check(self.frontend.vision,())
            self.graph.replay();grids.append(self.output.clone())
        return torch.stack(grids)
