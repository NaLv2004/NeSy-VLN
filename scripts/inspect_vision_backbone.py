"""Read-only inventory of already downloaded vision weights and their local API."""
import inspect,json,sys
from pathlib import Path
import transformers
root=Path('/home/user/xjy/nesy_vla_20261010')
cfg=json.loads((root/'symbolic_robust_v7/configs/robust_reference.json').read_text())
model=Path(cfg['model_path']);config=json.loads((model/'config.json').read_text())
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
files=[]
for path in model.glob('*.index.json'):
    data=json.loads(path.read_text());weights=data.get('weight_map',{})
    selected={k:v for k,v in weights.items() if any(x in k for x in ['visual.','vision_model.'])}
    files.append({'index':str(path),'vision_tensor_count':len(selected),'examples':list(selected.items())[:8]})
result={'transformers':transformers.__version__,'model_path':str(model),'vision_config':config.get('vision_config'),
    'vision_forward_signature':str(inspect.signature(Qwen3_5VisionModel.forward)),
    'vision_forward_source':inspect.getsource(Qwen3_5VisionModel.forward),
    'weight_inventory':files,'processor_files':[str(p) for p in model.glob('*processor*')]}
out=root/'diagnostics_20261011/vision_backbone_inventory.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='vision_forward_source'},indent=2))
