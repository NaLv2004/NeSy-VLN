"""Training-only paired feature probe; no policy or navigation action output.

Original Qwen weights only. The fixed episode split matches the earlier visual
probe. Images, action labels and generated state targets never include dev/final.
Both learned probes below are diagnostics, never runtime action selectors.
"""
import fcntl, hashlib, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_qwen_features_v47'
BASE=Path('/home/user/xjy/qwen3_5_2b_20260920/model')

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    OUT.mkdir(exist_ok=True);lock=(OUT/'job.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'report.json').exists(), 'Do not repeat completed probe'
    torch.set_num_threads(3);torch.manual_seed(471017)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=True
    began=time.time()
    def status(stage,**kw):
        value=dict(stage=stage,pid=os.getpid(),time=time.time(),seconds=time.time()-began,**kw)
        atomic(OUT/'status.json',value);print(json.dumps(value),flush=True)
    status('loading')
    source_dataset=ROOT/'diagnostic_dinov2_v43/dataset.json'
    ds=json.loads(source_dataset.read_text())
    manifest=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
    row_file=Path(manifest['data']['source'])/'training_rows.json'
    assert hashlib.sha256(row_file.read_bytes()).hexdigest()==ds['source_sha256']
    rows=json.loads(row_file.read_text());indices=ds['source_row_indices'];records=ds['records']
    import re
    for i in indices:
        row=rows[i];origin=row.get('source_episode_id',row['episode_id'])
        layout='layout_'+re.search(r'P25_train_(\d+)_',origin)[1]
        assert layout in manifest['data']['train_layouts']
    lookup={v:i for i,v in enumerate(indices)}
    fit_rows=sorted({lookup[r[k]] for r in records if r['split']=='fit' for k in ('current','successor')})
    assert not set(ds['fit_episode_ids'])&set(ds['heldout_episode_ids'])
    atomic(OUT/'manifest.json',dict(dataset_sha256=hashlib.sha256(source_dataset.read_bytes()).hexdigest(),
        dataset=ds,base=str(BASE),base_config_sha256=hashlib.sha256((BASE/'config.json').read_bytes()).hexdigest(),
        pretrained_base_only=True,adapters=[],image_size=[224,224],feature_budget=[49,128],
        fit_projection_only_on_fit_episodes=True,runtime_policy_modified=False,
        prompt='Observe this image.',goal_or_action_language_given=False))
    processor=AutoProcessor.from_pretrained(BASE,local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    vision_capture=[]
    def capture(_module,_args,result):
        vision_capture.append(result.pooler_output.detach().float().cpu())
    handle=model.model.visual.register_forward_hook(capture)
    vision=[];body=[]
    def load(index):
        with Image.open(rows[index]['path']) as image:
            return image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for begin in range(0,len(indices),8):
            images=list(pool.map(load,indices[begin:begin+8]))
            messages=[[dict(role='user',content=[dict(type='image',image=image),
                       dict(type='text',text='Observe this image.')])] for image in images]
            batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt',padding=True,enable_thinking=False)
            batch={k:v.cuda() for k,v in batch.items()}
            vision_capture.clear()
            result=model.model(**batch,use_cache=False,return_dict=True)
            mask=batch['input_ids']==model.config.image_token_id
            assert (mask.sum(-1)==49).all(),mask.sum(-1)
            tokens=result.last_hidden_state[mask].reshape(len(images),49,2048)
            assert len(vision_capture)==1
            vision.append(vision_capture[0].reshape(len(images),49,2048))
            body.append(tokens.float().cpu())
            if begin==0 or begin%256==0:
                status('encoding',images=min(begin+8,len(indices)),total=len(indices))
    handle.remove();del model,processor,result,batch,tokens;torch.cuda.empty_cache()
    encoded={'qwen_original_vision':torch.cat(vision),'qwen_original_multimodal':torch.cat(body)}
    del vision,body
    projected={}
    for name,features in encoded.items():
        status('projection',encoder=name)
        # Covariance uses only patches from fitting episodes. Held-out episodes
        # do not determine the normalization or PCA axes.
        patch=features[fit_rows].reshape(-1,2048)
        patch=patch[torch.randperm(len(patch))[:min(20000,len(patch))]].cuda()
        mu=patch.mean(0);sd=patch.std(0).clamp_min(1e-4);standard=(patch-mu)/sd
        _,axes=torch.linalg.eigh(standard.T@standard/max(1,len(standard)-1));axes=axes[:,-128:]
        output=[]
        for begin in range(0,len(features),128):
            output.append(F.normalize(((features[begin:begin+128].cuda()-mu)/sd)@axes,dim=-1).cpu())
        projected[name]=torch.cat(output)
        torch.save(dict(mean=mu.cpu(),sd=sd.cpu(),axes=axes.cpu(),fit_episode_ids=ds['fit_episode_ids']),OUT/(name+'_projection.pt'))
        np.save(OUT/(name+'.npy'),projected[name].numpy().astype(np.float16))
        del patch,standard,axes,output;torch.cuda.empty_cache()
    del encoded
    raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    projected={'existing_convnext':torch.from_numpy(raw[indices].astype(np.float32)),**projected}
    action=torch.tensor([r['action'] for r in records],device='cuda')
    fit=torch.tensor([r['split']=='fit' for r in records],device='cuda')
    current=torch.tensor([lookup[r['current']] for r in records]);future=torch.tensor([lookup[r['successor']] for r in records])
    sys.path.insert(0,str(ROOT/'src_boundary_balanced_v43'))
    from model import FactoredWorld,Config
    saved=torch.load(ROOT/'diagnostic_sparse_support_v44/source.pt',map_location='cpu',weights_only=False)
    state_model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
    state_model.load_state_dict(saved['model'])
    with torch.no_grad():
        code=state_model.observe(projected['existing_convnext'].cuda()).argmax(-1)
    target_o=code[future.cuda()]
    del state_model,saved;torch.cuda.empty_cache()
    fit_indices=fit.nonzero()[:,0];results={}
    def scores(logits,labels,mask,classes):
        correct=logits.argmax(-1)==labels
        by_class={str(a):dict(n=int(((labels==a)&mask).sum()),accuracy=float(correct[(labels==a)&mask].float().mean()))
            for a in range(classes) if ((labels==a)&mask).any()}
        return dict(accuracy=float(correct[mask].float().mean()),samples=int(mask.sum()),
                    macro_accuracy=sum(v['accuracy'] for v in by_class.values())/len(by_class),by_class=by_class)
    for name,features in projected.items():
        results[name]={}
        for mode in ('inverse_dynamics','fixed_successor_code'):
            torch.manual_seed(431017)
            before=features[current].flatten(1)
            if mode=='inverse_dynamics':
                values=torch.cat([before,features[future].flatten(1)-before],-1).cuda();labels=action;classes=17
            else:
                values=torch.cat([before.cuda(),F.one_hot(action,17).float()],-1);labels=target_o;classes=64
            mu=values[fit].mean(0);sd=values[fit].std(0).clamp_min(.01);values=(values-mu)/sd
            head=nn.Sequential(nn.Linear(values.shape[-1],128),nn.SiLU(),nn.Linear(128,classes)).cuda()
            optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
            for update in range(500):
                chosen=fit_indices[torch.randint(len(fit_indices),(min(256,len(fit_indices)),),device='cuda')]
                loss=F.cross_entropy(head(values[chosen]),labels[chosen]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():
                logits=head(values);row=dict(fit=scores(logits,labels,fit,classes),heldout=scores(logits,labels,~fit,classes))
            results[name][mode]=row
            status('probe_result',encoder=name,mode=mode,heldout=row['heldout']['accuracy'],heldout_macro=row['heldout']['macro_accuracy'])
            del head,optimizer,values;torch.cuda.empty_cache()
    report=dict(training_only=True,results=results,seconds=time.time()-began,frame_count=len(indices),
        fit_episodes=len(ds['fit_episode_ids']),heldout_episodes=len(ds['heldout_episode_ids']),
        peak_gib=torch.cuda.max_memory_allocated()/2**30,runtime_action_head_installed=False,
        limitations=['Episode holdout applies to the new diagnostic heads and Qwen PCA. Existing ConvNeXt and source discrete encoder previously saw the broader training corpus.',
                    'Fixed successor labels deliberately remain the v43 source labels; this metric does not prove a better learned symbol system.',
                    'Inverse dynamics measures recoverable action evidence, not physical navigation success. No dev/final episodes evaluated.'])
    atomic(OUT/'report.json',report);status('complete')

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():atomic(OUT/'error.json',dict(time=time.time(),pid=os.getpid(),error=repr(error)))
        raise
