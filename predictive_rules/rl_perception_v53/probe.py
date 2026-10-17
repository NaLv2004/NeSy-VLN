"""Probe prompt-only RL perception, never an action-generating runtime model.

Same images, fixed split and heads as v47. Put the task BEFORE image tokens so
causal image hidden states can attend to it. Paired disabled/enabled LoRA passes
separate this prompt change from actual navigation finetuning. No generated or
teacher answer token is ever supplied to the feature encoder.
"""
import fcntl,hashlib,json,os,re,sys,time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_rl_perception_v53'
BASE=Path('/home/user/xjy/qwen3_5_2b_20260920/model')
ADAPTER=Path('/home/user/xjy/lite3_photo_world_20260927/lab_rl64/run32_v4_photo25/step_000249/adapter')
EXPECTED='cbaa6dac9e65df3b9a62759087c86c958f46af3e85e53b44309242867d014e3b'

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    OUT.mkdir(exist_ok=True);lock=(OUT/'job.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'report.json').exists(),'Completed probe must not be repeated'
    torch.set_num_threads(3);torch.manual_seed(531017)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=True
    began=time.time()
    def status(stage,**kw):
        row=dict(stage=stage,pid=os.getpid(),time=time.time(),seconds=time.time()-began,**kw)
        atomic(OUT/'status.json',row);print(json.dumps(row),flush=True)
    status('loading')
    ds_path=ROOT/'diagnostic_dinov2_v43/dataset.json';ds=json.loads(ds_path.read_text())
    manifest=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
    row_path=Path(manifest['data']['source'])/'training_rows.json'
    assert hashlib.sha256(row_path.read_bytes()).hexdigest()==ds['source_sha256']
    rows=json.loads(row_path.read_text());indices=ds['source_row_indices'];records=ds['records']
    for i in indices:
        row=rows[i];origin=row.get('source_episode_id',row['episode_id'])
        layout='layout_'+re.search(r'P25_train_(\d+)_',origin)[1]
        assert layout in manifest['data']['train_layouts']
        assert row['category'] in ('chair','table')
    assert not set(ds['fit_episode_ids'])&set(ds['heldout_episode_ids'])
    assert (ADAPTER.parent/'COMMITTED').exists()
    adapter_sha=hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert adapter_sha==EXPECTED
    lookup={v:i for i,v in enumerate(indices)}
    fit_rows=sorted({lookup[r[k]] for r in records if r['split']=='fit' for k in ('current','successor')})
    prompt='Task: Find a {category}. Observe the current image.'
    atomic(OUT/'manifest.json',dict(training_only=True,source_dataset_sha256=hashlib.sha256(ds_path.read_bytes()).hexdigest(),
        source_dataset=str(ds_path),base=str(BASE),adapter=str(ADAPTER),adapter_sha256=adapter_sha,
        fit_episode_ids=ds['fit_episode_ids'],heldout_episode_ids=ds['heldout_episode_ids'],
        image_size=[224,224],feature_budget=[49,128],prompt_before_image=prompt,
        no_generated_tokens=True,no_teacher_answer_tokens=True,no_runtime_action_head=True,
        paired_base_uses_same_prompt=True,projection_fit_only_on_fitting_episodes=True,
        known_limitation='RL teacher and ConvNeXt saw broader training scenes; heldout split tests new heads, not pretraining independence.'))
    processor=AutoProcessor.from_pretrained(BASE,local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    base=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False).cuda().eval().requires_grad_(False)
    body=model.get_base_model().model
    forbidden_calls=[]
    def forbidden(*_):
        forbidden_calls.append('lm_head');raise RuntimeError('Vocabulary output is forbidden in this perception probe')
    handle=model.get_output_embeddings().register_forward_pre_hook(forbidden)
    for p in model.parameters():assert not p.requires_grad
    buffers={name:[] for name in ('base_task_prefix','rl_task_prefix')}
    def load(index):
        with Image.open(rows[index]['path']) as image:
            return image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    causality={}
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for begin in range(0,len(indices),8):
            selected=indices[begin:begin+8];images=list(pool.map(load,selected))
            messages=[[dict(role='user',content=[dict(type='text',text=prompt.format(category=rows[i]['category'])),
                dict(type='image',image=image)])] for i,image in zip(selected,images)]
            batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,
                return_dict=True,return_tensors='pt',padding=True,enable_thinking=False)
            batch={k:v.cuda() for k,v in batch.items()}
            mask=batch['input_ids']==base.config.image_token_id
            assert (mask.sum(-1)==49).all()
            for name in buffers:
                with model.disable_adapter() if name=='base_task_prefix' else nullcontext():
                    result=body(**batch,use_cache=False,return_dict=True)
                    hidden=result.last_hidden_state[mask].reshape(len(images),49,2048).float().cpu()
                    buffers[name].append(hidden)
            if begin==0:
                first=processor.tokenizer.decode(batch['input_ids'][0].tolist(),skip_special_tokens=False)
                assert first.index('Task:')<first.index('<|image_pad|>')
                causality.update(first_prompt_tokens=first,lm_head_calls=len(forbidden_calls),
                    paired_rl_base_feature_rms=float((buffers['base_task_prefix'][0]-buffers['rl_task_prefix'][0]).square().mean().sqrt()),
                    adapter_layers=sum(1 for n,_ in body.named_modules() if 'lora_A.default' in n))
                assert causality['adapter_layers']>0
                atomic(OUT/'feature_contract.json',causality)
            if begin==0 or begin%256==0:status('encoding',images=min(begin+8,len(indices)),total=len(indices))
    assert not forbidden_calls;handle.remove()
    del model,body,base,processor,result,batch;torch.cuda.empty_cache()
    projected={}
    for name,chunks in buffers.items():
        status('projection',encoder=name)
        features=torch.cat(chunks);chunks.clear()
        patch=features[fit_rows].reshape(-1,2048)
        # Identical random sample locations for the paired projections.
        generator=torch.Generator().manual_seed(471017)
        patch=patch[torch.randperm(len(patch),generator=generator)[:min(20000,len(patch))]].cuda()
        mu=patch.mean(0);sd=patch.std(0).clamp_min(1e-4);standard=(patch-mu)/sd
        _,axes=torch.linalg.eigh(standard.T@standard/max(1,len(standard)-1));axes=axes[:,-128:]
        values=[]
        for begin in range(0,len(features),128):
            values.append(F.normalize(((features[begin:begin+128].cuda()-mu)/sd)@axes,dim=-1).cpu())
        projected[name]=torch.cat(values)
        if name=='rl_task_prefix':
            torch.save(dict(mean=mu.cpu(),sd=sd.cpu(),axes=axes.cpu(),fit_episode_ids=ds['fit_episode_ids']),OUT/'projection.pt')
            np.save(OUT/'rl_task_prefix.npy',projected[name].numpy().astype(np.float16))
        del patch,standard,features,values;torch.cuda.empty_cache()
    del buffers
    raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    projected={'existing_convnext':torch.from_numpy(raw[indices].astype(np.float32)),**projected}
    for name in ('qwen_original_vision','qwen_original_multimodal'):
        projected[name]=torch.from_numpy(np.load(ROOT/'diagnostic_qwen_features_v47'/(name+'.npy')).astype(np.float32))
    action=torch.tensor([r['action'] for r in records],device='cuda')
    fit=torch.tensor([r['split']=='fit' for r in records],device='cuda')
    current=torch.tensor([lookup[r['current']] for r in records]);future=torch.tensor([lookup[r['successor']] for r in records])
    sys.path.insert(0,str(ROOT/'src_boundary_balanced_v43'))
    from model import FactoredWorld,Config
    saved=torch.load(ROOT/'diagnostic_sparse_support_v44/source.pt',map_location='cpu',weights_only=False)
    state_model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
    state_model.load_state_dict(saved['model'])
    with torch.no_grad():code=state_model.observe(projected['existing_convnext'].cuda()).argmax(-1)
    target_o=code[future.cuda()];del state_model,saved;torch.cuda.empty_cache()
    fit_indices=fit.nonzero()[:,0];results={}
    tasks=F.one_hot(torch.tensor([['chair','table'].index(rows[r['current']]['category']) for r in records],device='cuda'),2).float()
    def scores(logits,labels,mask,classes):
        correct=logits.argmax(-1)==labels
        by_class={str(a):dict(n=int(((labels==a)&mask).sum()),accuracy=float(correct[(labels==a)&mask].float().mean()))
            for a in range(classes) if ((labels==a)&mask).any()}
        return dict(accuracy=float(correct[mask].float().mean()),samples=int(mask.sum()),
            macro_accuracy=sum(v['accuracy'] for v in by_class.values())/len(by_class),by_class=by_class)
    for name,features in projected.items():
        results[name]={}
        # Historical modes reuse v47 output for unchanged caches; only the new
        # current-view diagnostic requires fresh heads on these existing codes.
        old=json.loads((ROOT/'diagnostic_qwen_features_v47/report.json').read_text())['results']
        for mode in ('inverse_dynamics','fixed_successor_code','current_teacher_action'):
            if name in old and mode in old[name]:
                results[name][mode]={**old[name][mode],'reused_completed_v47':True};continue
            torch.manual_seed(431017)
            before=features[current].flatten(1)
            if mode=='inverse_dynamics':
                values=torch.cat([before,features[future].flatten(1)-before],-1).cuda();labels=action;classes=17
            elif mode=='fixed_successor_code':
                values=torch.cat([before.cuda(),F.one_hot(action,17).float()],-1);labels=target_o;classes=64
            else:
                values=torch.cat([before.cuda(),tasks],-1);labels=action;classes=17
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
    report=dict(training_only=True,results=results,feature_contract=causality,seconds=time.time()-began,
        frame_count=len(indices),fit_episodes=len(ds['fit_episode_ids']),heldout_episodes=len(ds['heldout_episode_ids']),
        peak_gib=torch.cuda.max_memory_allocated()/2**30,runtime_action_head_installed=False,
        limitations=['All heads are diagnostic and discarded, not runtime policies.',
            'Episode holdout applies to heads/projection. RL teacher and existing frontend had broader training exposure.',
            'Fixed successor labels remain old v43 codes; predictive probe is not a new symbolic model.',
            'Current-view teacher action prediction may be ambiguous without full history; it is not physical success.',
            'No dev or finalfixed episodes used. All policy execution and training remain unchanged.'])
    atomic(OUT/'report.json',report);status('complete')

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():atomic(OUT/'error.json',dict(time=time.time(),pid=os.getpid(),error=repr(error)))
        raise
