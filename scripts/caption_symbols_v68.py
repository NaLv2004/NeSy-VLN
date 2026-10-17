"""Resumable image-to-discrete-text frontend; no policy or task labels in prompts."""
import argparse,fcntl,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration

B=Path('/home/user/xjy/nesy_vla_20261010')
sys.path.insert(0,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import atomic,read,sha,setup,DataBank
PROMPT='Describe this image precisely enough to distinguish it from other views of the same room. State only what can be observed. Use at most 70 words.'

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--pilot',action='store_true')
 p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
 p.add_argument('--pixels',type=int,default=448*448);p.add_argument('--batch',type=int,default=16)
 args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 lock=(out/f'rank{args.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 result=out/f'rank{args.rank}.result.json';assert not result.exists(),'Completed shard cannot be repeated'
 status={'pid':os.getpid(),'rank':args.rank,'state':'initializing','time':time.time()}
 def progress(**kw):
  status.update(kw,time=time.time());atomic(out/f'rank{args.rank}.status.json',status);print(json.dumps(status),flush=True)
 try:
  torch.set_num_threads(2);torch.manual_seed(6800)
  cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
  rows_path=B/'visual_vocabulary_probe_v53/rows.json';rows=read(rows_path)
  if args.pilot:rows=[a['row'] for a in read(B/'semantic_tokens_v67/pilot_result.json')['answers']]
  assert all(bank.by_id[r['episode_id']]['scene'] in bank.provenance['train_layouts'] for r in rows)
  unique={}
  for r in rows:unique.setdefault(sha(r['path']),r['path'])
  targets=[{'image_sha256':h,'image':path} for h,path in sorted(unique.items())][args.rank::args.world]
  model_path=cfg['model_path']
  index=read(Path(model_path)/'model.safetensors.index.json')
  weight_files=sorted(set(index['weight_map'].values()))
  # Hashing existing immutable weights is read-only; no model copies are made.
  frontend={'model':model_path,'weight_sha256':{n:sha(Path(model_path)/n) for n in weight_files},
   'config_sha256':sha(Path(model_path)/'config.json'),'prompt':PROMPT,'pixels':args.pixels,'max_new_tokens':96,
   'do_sample':False,'enable_thinking':False,'output':'Emitted integer BPE token IDs only; no image or VLM hidden features cross into reasoning.'}
  identity=hashlib.sha256(json.dumps(frontend,sort_keys=True).encode()).hexdigest()
  manifest={'frontend':frontend,'frontend_sha256':identity,'source_rows_sha256':sha(rows_path),'rank':args.rank,
   'world':args.world,'batch':args.batch,'pilot':args.pilot,'targets':targets,'official_heldouts_used':False}
  mpath=out/f'rank{args.rank}.manifest.json'
  if mpath.exists():assert read(mpath)==manifest,'Resume configuration changed'
  else:atomic(mpath,manifest)
  ledger=out/f'rank{args.rank}.jsonl';completed={}
  if ledger.exists():
   raw=ledger.read_bytes();lines=raw.splitlines(keepends=True);valid=0
   for line in lines:
    if not line.endswith(b'\n'):break
    item=json.loads(line);assert item['frontend_sha256']==identity
    assert item['image_sha256'] not in completed;completed[item['image_sha256']]=item;valid+=len(line)
   # A killed process may leave only the last append incomplete.
   if valid!=len(raw):
    with ledger.open('r+b') as f:f.truncate(valid)
  pilot=B/'semantic_tokens_v68/pilot/rank0.jsonl'
  if not args.pilot and pilot.exists():
   wanted={r['image_sha256'] for r in targets}
   with ledger.open('a') as f:
    for line in pilot.read_text().splitlines():
     item=json.loads(line)
     if item['frontend_sha256']==identity and item['image_sha256'] in wanted and item['image_sha256'] not in completed:
      f.write(json.dumps(item)+'\n');completed[item['image_sha256']]=item
    f.flush();os.fsync(f.fileno())
  remaining=[r for r in targets if r['image_sha256'] not in completed]
  if remaining:
   progress(state='loading',completed=len(completed),total=len(targets))
   processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=args.pixels,max_pixels=args.pixels)
   processor.tokenizer.padding_side='left'
   model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,
    attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
   assert sum(p.numel() for p in model.parameters())<3_000_000_000
   model.generation_config.disable_compile=True
   eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
   started=time.time()
   offset=0;active_batch=args.batch
   while offset<len(remaining):
    batch=remaining[offset:offset+active_batch];images=[Image.open(r['image']).convert('RGB') for r in batch]
    conversations=[[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':PROMPT}]}] for im in images]
    inputs=processor.apply_chat_template(conversations,tokenize=True,padding=True,add_generation_prompt=True,
     enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda');width=inputs['input_ids'].shape[1]
    try:
     with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=96,do_sample=False,use_cache=True)
    except torch.cuda.OutOfMemoryError:
     if active_batch==1:raise
     del inputs;gc.collect();torch.cuda.empty_cache();active_batch=max(1,active_batch//2)
     progress(state='reducing_batch_after_oom',active_batch=active_batch,completed=len(completed),total=len(targets));continue
    with ledger.open('a') as f:
     for row,ids in zip(batch,outputs[:,width:].tolist()):
      end=next((i for i,t in enumerate(ids) if t in eos),None);tokens=ids[:end] if end is not None else ids
      item=dict(row,frontend_sha256=identity,token_ids=tokens,caption=processor.tokenizer.decode(tokens,skip_special_tokens=True),
       terminated=end is not None,decoded_tokens=len(tokens))
      f.write(json.dumps(item)+'\n');completed[row['image_sha256']]=item
     f.flush();os.fsync(f.fileno())
    offset+=len(batch)
    progress(state='captioning',completed=len(completed),total=len(targets),seconds=time.time()-started,
     active_batch=active_batch,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
  atomic(result,{'completed':len(completed),'total':len(targets),'frontend_sha256':identity,'time':time.time(),
   'truncated':sum(not r['terminated'] for r in completed.values()),'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
  progress(state='complete',completed=len(completed),total=len(targets))
 except BaseException as e:progress(state='failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
