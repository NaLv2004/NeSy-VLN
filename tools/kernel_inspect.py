import importlib.util,inspect,json
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as q
print('torch',torch.__version__,'module',q.__file__)
for name in ['fla','causal_conv1d','triton','einops']:
    spec=importlib.util.find_spec(name);print(name,None if spec is None else spec.origin)
lines=inspect.getsource(q).splitlines()
for i,line in enumerate(lines):
    if 'chunk_gated_delta_rule' in line or 'is_fast_path' in line or 'fla.' in line:
        print(i+1,'\n'.join(lines[max(0,i-1):i+3]))
