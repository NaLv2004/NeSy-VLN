"""Inspect automatically generated visual descriptions before a larger study.

The frozen VLM sees an image and a generic description request only. It receives
no action labels, goal geometry, maps or hand-specified predicate vocabulary.
"""
import fcntl,json,os,sys,time,traceback
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'semantic_tokens_v67';out.mkdir(exist_ok=True)
sys.path.insert(0,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,atomic,sha,setup,DataBank
lock=(out/'pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'pilot_result.json').exists(),'Completed pilot cannot be repeated'
cfg=read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
rows=read(b/'visual_vocabulary_probe_v53/rows.json');selected=[];seen=set()
# Labels select diverse training examples for inspection; they are never
# supplied to the VLM. Official development and fixed80 are excluded.
for category in ['chair','table']:
 for label in [0,1,2,3,4,5,6,7]:
  candidates=[r for r in rows if r['category']==category and r['label']==label and r['supervised'] and r['probe_split']=='fit']
  if not candidates:continue
  row=next((r for r in candidates if sha(r['path']) not in seen),None)
  if row is None:continue
  digest=sha(row['path']);seen.add(digest);selected.append(dict(row,image_sha256=digest))
assert len(selected)>=12
assert all(bank.by_id[r['episode_id']]['scene'] in bank.provenance['train_layouts'] for r in selected)
prompt='Describe this image precisely enough to distinguish it from other views of the same room. State only what can be observed. Use at most 70 words.'
path=cfg['model_path'];torch.manual_seed(6700);torch.set_num_threads(4)
state={'pid':os.getpid(),'state':'loading','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'pilot_status.json',state);print(json.dumps(state),flush=True)
try:
 processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=224*224,max_pixels=224*224)
 processor.tokenizer.padding_side='left'
 model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
 model.generation_config.disable_compile=True
 meta={'model':path,'parameters':sum(p.numel() for p in model.parameters()),'prompt':prompt,'thinking':False,
  'max_new_tokens':96,'do_sample':False,'image_min_pixels':224*224,'image_max_pixels':224*224,'batch':4,
  'source_rows_sha256':sha(b/'visual_vocabulary_probe_v53/rows.json'),'official_heldouts_used':False,
  'outputs':'Only generated text and integer BPE IDs. No VLM hidden/image features will enter the symbolic program learner.'}
 assert meta['parameters']<3_000_000_000
 atomic(out/'pilot_manifest.json',meta);answers=[];started=time.time()
 for offset in range(0,len(selected),4):
  batch=selected[offset:offset+4];images=[Image.open(r['path']).convert('RGB') for r in batch]
  conversations=[[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}] for image in images]
  inputs=processor.apply_chat_template(conversations,tokenize=True,padding=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda')
  width=inputs['input_ids'].shape[1]
  with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=96,do_sample=False,use_cache=True)
  eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  for row,ids,image in zip(batch,generated[:,width:].tolist(),images):
   terminated=next((i for i,t in enumerate(ids) if t in eos),None)
   tokens=ids[:terminated] if terminated is not None else ids
   answer={'row':row,'caption':processor.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,
    'terminated':terminated is not None,'original_image_size':image.size,'decoded_tokens':len(tokens)}
   answers.append(answer)
  atomic(out/'pilot_partial.json',{'manifest':meta,'answers':answers})
  progress(state='captioning',images=len(answers),total=len(selected),seconds=time.time()-started)
 atomic(out/'pilot_result.json',{'manifest':meta,'answers':answers,'seconds':time.time()-started,'finished':time.time()})
 progress(state='complete',images=len(answers))
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
