"""Encode ONLY emitted integer text tokens with the frozen language model.

This second pass receives no pixels, visual features, image tokens, task or
action labels. Its cache is keyed by complete emitted text-token sequences.
"""
import fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
import torch
from transformers import Qwen3_5ForConditionalGeneration
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'text_context_v70';out.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists()
state={'pid':os.getpid(),'state':'loading','time':time.time()}
def progress(**kw):
 state.update(kw,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
try:
 torch.set_num_threads(2);torch.manual_seed(7000);corpus=B/'semantic_tokens_v68/corpus'
 identity=read(corpus/'rank0.manifest.json')['frontend_sha256'];frontend=read(corpus/'rank0.manifest.json')['frontend'];sequences=set()
 ledger_hashes={}
 for rank in range(8):
  done=read(corpus/f'rank{rank}.result.json');assert done['completed']==done['total']
  ledger_hashes[str(rank)]=sha(corpus/f'rank{rank}.jsonl')
  for line in (corpus/f'rank{rank}.jsonl').read_text().splitlines():
   item=json.loads(line);assert item['frontend_sha256']==identity;sequences.add(tuple(item['token_ids']))
 sequences=sorted(sequences);path=frontend['model']
 full=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
 language=full.model.language_model;del full
 language=language.cuda().eval().requires_grad_(False)
 assert sum(p.numel() for p in language.parameters())<3_000_000_000
 started=time.time();vectors=[];bs=64
 with torch.inference_mode():
  for offset in range(0,len(sequences),bs):
   batch=sequences[offset:offset+bs];tokens=torch.zeros((len(batch),96),dtype=torch.long,device='cuda')
   mask=torch.zeros_like(tokens)
   for i,seq in enumerate(batch):tokens[i,:len(seq)]=torch.tensor(seq,device='cuda');mask[i,:len(seq)]=1
   # Deliberately invoke the text tower, not the multimodal wrapper.
   output=language(input_ids=tokens,attention_mask=mask,use_cache=False).last_hidden_state
   pooled=(output.float()*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
   vectors.append(pooled.to(torch.bfloat16).cpu())
   progress(state='encoding_text_only',completed=min(offset+bs,len(sequences)),total=len(sequences),seconds=time.time()-started,
    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
 cache={'sequences':sequences,'vectors':torch.cat(vectors),'frontend_sha256':identity,
  'text_model':path,'weight_sha256':frontend['weight_sha256'],'caption_ledgers_sha256':ledger_hashes,
  'construction':'Frozen text tower only; attention-mask mean pooling. No images or multimodal hidden states supplied.'}
 target=out/'text_only_cache.pt';temporary=out/'text_only_cache.partial.pt';torch.save(cache,temporary);os.replace(temporary,target)
 result={'time':time.time(),'sequences':len(sequences),'width':cache['vectors'].shape[-1],'cache':str(target),'cache_sha256':sha(target),
  'seconds':time.time()-started,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'frontend_sha256':identity}
 atomic(out/'result.json',result);progress(state='complete',**result)
except BaseException as e:progress(state='failed',error=repr(e),traceback=traceback.format_exc());raise
