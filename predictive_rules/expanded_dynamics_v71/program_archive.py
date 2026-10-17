"""Lossless program recipes: immutable weights, code, and generation inputs.

The live interpreter still receives the full compiled discrete table. This
module changes disk storage only. Loading a recipe recompiles the table and
requires its complete byte-level digest to match the executed table.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot(model,out,rank):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model_source=Path(sys.modules[type(model).__module__].__file__)
    source_bytes=model_source.read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest()
    code=out/f'model_{source_sha}.py'
    if not code.exists():
        temp=code.with_suffix(f'.rank{rank}.tmp');temp.write_bytes(source_bytes);temp.replace(code)
    destination=out/f'program_model_rank{rank}.pt'
    temp=destination.with_suffix('.tmp')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        model_config=model.export_config(),source_file=code.name,source_sha256=source_sha,
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(next(model.parameters()).device),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),temp)
    temp.replace(destination)
    return dict(file=destination.name,sha256=sha(destination),format='exact_program_recipe_v1')

def save_recipe(path,context,task,program,archive):
    path=Path(path);temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,context=context.detach().cpu().numpy(),task=task.detach().cpu().numpy(),
            program_sha256=np.asarray(program.digest()),model_sha256=np.asarray(archive['sha256']),
            model_file=np.asarray(archive['file']),observations=np.asarray(program.observations))
    temp.replace(path)

@torch.no_grad()
def restore(path,device='cuda'):
    from planner import Program
    path=Path(path)
    with np.load(path,allow_pickle=False) as seed:
        out=path.parent.parent
        model_file=out/str(seed['model_file']);assert model_file.parent==out
        assert sha(model_file)==str(seed['model_sha256'])
        saved=torch.load(model_file,map_location='cpu',weights_only=False)
        code=out/saved['source_file'];assert code.parent==out and sha(code)==saved['source_sha256']
        name='_archived_rules_'+saved['source_sha256']
        spec=importlib.util.spec_from_file_location(name,code)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        model=module.FactoredWorld(module.Config(**saved['model_config'])).to(device).eval()
        model.load_state_dict(saved['model'])
        old_flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=saved['matmul_tf32']
            torch.backends.cudnn.allow_tf32=saved['cudnn_tf32']
            exported=model.compile(torch.from_numpy(seed['context'].copy()).to(device),
                torch.from_numpy(seed['task'].copy()).to(device))
        finally:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old_flags
        program=Program(**exported,observations=int(seed['observations'])).validate()
        assert program.digest()==str(seed['program_sha256']),'Recompiled rule table differs from executed table'
        return program
