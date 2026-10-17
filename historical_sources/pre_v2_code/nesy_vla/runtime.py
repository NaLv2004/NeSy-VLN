import hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path,data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
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
    sys.path.insert(0,str(Path(cfg['root'])/'deps'))
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
        demo_file=self.root/'data/search_cost_supervision_v2/examples.json'
        for row in read(demo_file):
            ep=self.by_id.get(row['episode_id'])
            if ep is None or ep['scene'] not in train_scenes:raise ValueError('Demonstration split contamination')
            self.demos.setdefault(row['episode_id'],[]).append(row)
        for rows in self.demos.values():
            rows.sort(key=lambda x:x['image'])
            assert rows[-1]['action']==0
        self.demo_ids=list(self.demos)
        self.provenance={'train_hash':sha(self.source/'train.json'),'fixed_hash':sha(self.source/'fixed80.json'),
            'demonstration_hash':sha(demo_file),'train_layouts':sorted(train_scenes),
            'development_layouts':self.lock['development_scenes'],'fixed_layouts':self.lock['fixed_scenes'],
            'demonstrations':len(self.demos),'demo_steps':sum(map(len,self.demos.values()))}

    def sample(self,rng,index):
        group=('dense' if index%4==3 else 'normal',['chair','table'][(index//4+index)%2],
            ['near','medium','medium','far','far'][index%5])
        return rng.choice(self.groups[group])

    def simulator(self,log,gpu,evaluation=False):
        from stripsnav.sim_client import Simulator
        return Simulator(self.env,log,evaluation=evaluation,gpu_index=gpu)


def save_checkpoint(folder,model,optimizer,update,cfg,rng_states):
    from stripsnav.io import committed_torch_save
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    buffers={n for n,_ in model.named_buffers() if not n.startswith('generator.vlm.')}
    state={n:p.detach().cpu() for n,p in model.state_dict().items()
           if n in trainable or n in buffers or n.startswith('teacher.')}
    committed_torch_save(folder,{'model':state,'optimizer':optimizer.state_dict(),'update':update,
        'config':cfg,'rng_states':rng_states,'architecture':'TCDPP_full_v1'},
        {'epoch':update,'revision_interval':cfg['revision_interval'],'horizon':cfg['horizon']})


def load_checkpoint(folder,model,optimizer=None):
    from stripsnav.io import load_committed
    state=load_committed(folder)
    result=model.load_state_dict(state['model'],strict=False)
    missing=set(result.missing_keys)
    assert not [n for n,p in model.named_parameters() if p.requires_grad and n in missing]
    assert not result.unexpected_keys
    if optimizer is not None:optimizer.load_state_dict(state['optimizer'])
    return state
