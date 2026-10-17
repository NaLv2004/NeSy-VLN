import hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path,data,compact=False):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=None if compact else 2,
            separators=(',',':') if compact else None,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)


def append(path,data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f: f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def setup(cfg):
    sys.path.insert(0,cfg.get('dependency_root',str(Path(cfg['root'])/'deps')))
    sys.path.insert(0,cfg['reference_root'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True


def pil(rgb): return Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8))


def image_tensor(image,size,device):
    a=np.asarray(image.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32).copy()/255.
    t=torch.from_numpy(a).permute(2,0,1).to(device)[None]
    mean=t.new_tensor([.485,.456,.406])[None,:,None,None]
    std=t.new_tensor([.229,.224,.225])[None,:,None,None]
    return (t-mean)/std


def public_feedback(response,before,after):
    # Explicit allowlist. Reward, true distance, eligibility, pose and semantic masks never reach the VLM.
    raw=response.get('feedback',{})
    collision=bool(raw.get('collision',False))
    x=np.asarray(before.resize((32,32)),dtype=np.float32)/255
    y=np.asarray(after.resize((32,32)),dtype=np.float32)/255
    change=float(np.abs(x-y).mean())
    return {'collision':collision,'visual_change':round(change,5)}, [float(collision),change,float('collision' in raw)]


class DataBank:
    def __init__(self,cfg):
        self.root=Path(cfg['reference_root']); self.lock=read(self.root/'research/split_lock.json')
        self.env=read(self.root/'configs/experiment.json')
        self.env.update(training_horizon=cfg['horizon'],evaluation_horizon=cfg['horizon'])
        self.source=Path(self.env['source_data'])
        for name in ['train.json','fixed80.json']:
            assert sha(self.source/name)==self.lock['hashes'][name], 'Frozen split changed'
        all_tasks=read(self.source/'train.json'); self.by_id={x['episode_id']:x for x in all_tasks}
        train_scenes=set(self.lock['train_scenes'])-set(self.lock['development_scenes'])
        assert train_scenes.isdisjoint(self.lock['fixed_scenes'])
        self.train=[x for x in all_tasks if x['scene'] in train_scenes and x['distance_group']!='stop_ready']
        self.fixed=read(self.source/'fixed80.json')
        from stripsnav.development import development_tasks
        self.dev=development_tasks(all_tasks,self.lock['development_scenes'])
        self.groups={}
        for t in self.train:self.groups.setdefault((t['domain'],t['category'],t['distance_group']),[]).append(t)
        self.demos={}
        self.paired_demo_ids=[];self.paired_demo_groups=[];paired_provenance=[];self.recovery_demo_ids=[]
        demo_file=Path(cfg.get('demonstration_file',self.root/'data/search_cost_supervision_v2/examples.json'))
        self.demo_image_root=Path(cfg.get('demonstration_image_root',self.root/'data')).resolve()
        for row in read(demo_file):
            ep=self.by_id.get(row['episode_id'])
            if ep is None or ep['scene'] not in train_scenes:raise ValueError('Demonstration split contamination')
            self.demos.setdefault(row['episode_id'],[]).append(row)
        self.allowed_image_roots={self.demo_image_root}
        for paired_root in cfg.get('paired_demonstration_roots',[]):
            from .paired_data import load_paired_demonstrations
            extras,rows,evidence=load_paired_demonstrations(paired_root,self.by_id,train_scenes,self.lock['hashes'])
            self.by_id.update(extras);self.paired_demo_ids.extend(extras);paired_provenance.append(evidence)
            ordered=list(extras)
            self.paired_demo_groups.extend(tuple(ordered[i:i+2]) for i in range(0,len(ordered),2))
            self.allowed_image_roots.add(Path(paired_root).resolve())
            for row in rows:self.demos.setdefault(row['episode_id'],[]).append(row)
        recovery_provenance=None
        if cfg.get('recovery_demonstration_root'):
            from .recovery_data import load_recovery_demonstrations
            ids,rows,recovery_provenance=load_recovery_demonstrations(cfg['recovery_demonstration_root'],
                self.by_id,train_scenes,self.lock['hashes'],self.root/'data')
            if set(ids).intersection(self.demos):raise ValueError('Recovery episode overlaps existing demonstrations')
            self.recovery_demo_ids=ids;self.allowed_image_roots.add(Path(recovery_provenance['image_root']))
            for row in rows:self.demos.setdefault(row['episode_id'],[]).append(row)
        for rows in self.demos.values():
            rows.sort(key=lambda x:x['image'])
            assert rows[-1]['action']==0
        execution_provenance=None
        if cfg.get('execution_outcome_root'):
            from .outcome_supervision import attach_execution_outcomes
            execution_provenance=attach_execution_outcomes(cfg['execution_outcome_root'],self.by_id,self.demos,
                train_scenes,self.root/'data')
        self.demo_ids=list(self.demos)
        self.provenance={'train_hash':sha(self.source/'train.json'),'fixed_hash':sha(self.source/'fixed80.json'),
            'demonstration_hash':sha(demo_file),'demonstration_file':str(demo_file),
            'demonstration_image_root':str(self.demo_image_root),'train_layouts':sorted(train_scenes),
            'development_layouts':self.lock['development_scenes'],'fixed_layouts':self.lock['fixed_scenes'],
            'demonstrations':len(self.demos),'demo_steps':sum(map(len,self.demos.values()))}
        if paired_provenance:self.provenance['paired_goal_demonstrations']=paired_provenance
        if recovery_provenance:self.provenance['recovery_demonstrations']=recovery_provenance
        if execution_provenance:self.provenance['execution_outcomes']=execution_provenance

    def demo_image_path(self,row):
        root=Path(row.get('_image_root',self.demo_image_root)).resolve()
        if root not in self.allowed_image_roots:raise ValueError('Unregistered demonstration image root')
        path=(root/row['image']).resolve()
        if not path.is_relative_to(root):raise ValueError('Demonstration image escaped its declared dataset root')
        return path

    def sample(self,rng,index):
        group=('dense' if index%4==3 else 'normal',['chair','table'][(index//4+index)%2],
            ['near','medium','medium','far','far'][index%5])
        return rng.choice(self.groups[group])

    def simulator(self,log,gpu,evaluation=False):
        from stripsnav.sim_client import Simulator
        from .resources import physical_gpu
        return Simulator(self.env,log,evaluation=evaluation,gpu_index=physical_gpu(gpu))


def save_checkpoint(folder,model,optimizer,update,cfg,rng_states,include_optimizer=True):
    from stripsnav.io import committed_torch_save
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    buffers={n for n,_ in model.named_buffers() if not n.startswith('generator.vlm.')}
    frozen_vision=not any(p.requires_grad for p in model.grounder.vision.parameters())
    state={n:p.detach().cpu() for n,p in model.state_dict().items()
           if (n in trainable or n in buffers or n.startswith('teacher.')) and not
           (frozen_vision and n.startswith(('grounder.vision.','teacher.vision.')))}
    committed_torch_save(folder,{'model':state,'optimizer':optimizer.state_dict() if include_optimizer else None,'update':update,
        'config':cfg,'rng_states':rng_states,'architecture':cfg.get('architecture','TCDPP_full_v1')},
        {'epoch':update,'revision_interval':cfg['revision_interval'],'horizon':cfg['horizon'],
         'optimizer_included':include_optimizer})


def load_checkpoint(folder,model,optimizer=None):
    from stripsnav.io import load_committed
    state=load_committed(folder)
    result=model.load_state_dict(state['model'],strict=False)
    missing=set(result.missing_keys)
    assert not [n for n,p in model.named_parameters() if p.requires_grad and n in missing]
    assert not result.unexpected_keys
    if optimizer is not None:
        if state.get('optimizer') is None:raise ValueError('Evaluation-only checkpoint cannot resume optimizer')
        optimizer.load_state_dict(state['optimizer'])
    return state


def initialize_symbolic_modules(folder,model):
    """Transfer learned program/inference weights; new visual evidence is learned afresh.

    This is a new experiment with a fresh optimizer, never represented as an
    exact resume. No image-action teacher or hand-labelled predicate is used.
    """
    from stripsnav.io import load_committed
    state=load_committed(folder)
    prefixes=('generator.','logic.')
    selected={n:t for n,t in state['model'].items() if n.startswith(prefixes)}
    expected={n for n in model.state_dict() if n.startswith(prefixes)}
    assert selected.keys()==expected,'Symbolic initialization parameters differ'
    result=model.load_state_dict(selected,strict=False)
    assert not result.unexpected_keys and not any(n.startswith(prefixes) for n in result.missing_keys)
    return {'source':str(folder),'source_update':state['update'],'copied_modules':list(prefixes),
            'fresh_modules':['grounder','teacher'],'optimizer_restored':False}


def initialize_task_token_model(folder,model):
    """Preserve all learned components, adding only an observed-task token encoder.

    This is weight initialization with a fresh optimizer, not exact resumption.
    The vocabulary is derived from training episode instructions by the launcher.
    """
    from stripsnav.io import load_committed
    from .task_token_generator import TaskTokenProgramGenerator
    if not isinstance(model.generator,TaskTokenProgramGenerator):raise TypeError('Task-token generator required')
    state=load_committed(folder)
    if state['config'].get('generator_backend')!='symbolic_gru':raise ValueError('Expected symbolic GRU source')
    result=model.load_state_dict(state['model'],strict=False)
    expected={'generator.instruction_tokens.weight','generator.instruction_norm.weight','generator.instruction_norm.bias'}
    trainable_missing={n for n,p in model.named_parameters() if p.requires_grad and n in result.missing_keys}
    if trainable_missing!=expected or result.unexpected_keys:raise ValueError('Unexpected initialization mismatch')
    return {'source':str(folder),'source_update':state['update'],'optimizer_restored':False,
            'preserved_modules':['visual_grounder','EMA_teacher','rule_decoder','discrete_logic','symbolic_history'],
            'new_parameters':sorted(expected),'task_vocabulary':model.generator.task_vocabulary}


def initialize_execution_supervision(folder,model):
    """Transfer the policy exactly, adding explicit auxiliary/identity rule modules."""
    from stripsnav.io import load_committed
    if not model.cfg.get('execution_outcome_supervision') or not hasattr(model,'execution_decoder'):
        raise ValueError('Execution decoder must be explicitly enabled')
    state=load_committed(folder)
    if any(n.startswith('execution_decoder.') for n in state['model']):
        raise ValueError('Use ordinary checkpoint restoration for an existing decoder')
    result=model.load_state_dict(state['model'],strict=False)
    missing={n for n,p in model.named_parameters() if p.requires_grad and n in result.missing_keys}
    expected={n for n,p in model.named_parameters() if p.requires_grad and
        (n.startswith('execution_decoder.') or
         (n.startswith('logic.action_rule_gate.') and n not in state['model']))}
    if missing!=expected or result.unexpected_keys:raise ValueError('Unexpected execution initialization mismatch')
    return {'source':str(folder),'source_update':state['update'],'optimizer_restored':False,
            'all_existing_policy_parameters_preserved':True,
            'new_auxiliary_parameters':sorted(n for n in expected if n.startswith('execution_decoder.')),
            'new_identity_rule_parameters':sorted(n for n in expected if n.startswith('logic.action_rule_gate.')),
            'auxiliary_predictions_used_for_action_selection':False}
