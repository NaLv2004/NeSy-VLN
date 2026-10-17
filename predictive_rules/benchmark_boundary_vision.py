"""Training-image numerical and timing audit; no navigation evaluation or edits."""
import argparse,json,random,time,subprocess
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from PIL import Image
from predictive_nesy.environment import FrozenVision,load_previous
from predictive_nesy.data import read
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel,Config
from predictive_nesy.graph_subgoal_interpreter import export_program,plan

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);manifest=saved['manifest']
vision=FrozenVision(manifest['frozen_vision']);model=VisualGraphSubgoalModel(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
from nesy_vla.runtime import image_tensor
rows=read(Path(manifest['data']['source'])/'training_rows.json');episodes=defaultdict(list)
for row in rows:episodes[row['episode_id']].append(row)
selected=random.Random(431).sample([v for v in episodes.values() if len(v)>=21],8)
images=[[Image.open(r['path']).convert('RGB') for r in sample[:21]] for sample in selected]

@torch.no_grad()
def batched(frames):
    x=torch.cat([image_tensor(im,vision.cfg['image_size'],'cuda') for im in frames])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        f=F.adaptive_avg_pool2d(vision.vision(x),(7,7)).flatten(2).transpose(1,2)
    v=vision.vocabulary;z=F.normalize(((f.float()-v['mean'])/v['sd'])@v['axes'],dim=-1)
    return torch.cdist(z.flatten(0,1),v['centers']).argmin(1).reshape(len(frames),49)

@torch.no_grad()
def duration(function):
    torch.cuda.synchronize();begin=time.perf_counter();value=function();torch.cuda.synchronize();return time.perf_counter()-begin,value

with torch.no_grad():
    vision.encode(images[0][:11]);batched(images[0][:11]);torch.cuda.synchronize()
    serial=[];parallel=[];changed=0;total=0;program_equal=0;programs=0
    for frames in images:
        ts,cs=duration(lambda:vision.encode(frames));tb,cb=duration(lambda:batched(frames))
        serial.append(ts);parallel.append(tb);changed+=int((cs!=cb).sum());total+=cs.numel()
        # Same task and action-history tensors isolate the image-encoding change.
        task=torch.tensor([0],device='cuda');acts=torch.full((1,11),17,device='cuda');length=torch.tensor([11],device='cuda')
        ps=model.generate(task,cs[10:21][None],acts,length,start=20)
        pb=model.generate(task,cb[10:21][None],acts,length,start=20)
        programs+=1;program_equal+=int(all(torch.equal(ps[k],pb[k]) for k in ['definitions','transitions','subgoal_logits']))
    # A fixed replay measures computation only, not navigation accuracy.
    old,_,_=load_previous();old.eval();measurements=[]
    for sample,frames in zip(selected[:4],images[:4]):
        task=torch.tensor([['chair','table'].index(sample[0]['category'])],device='cuda')
        def reactive():
            history=[];_,memory=old.initial_state(1,'cuda');previous=torch.full_like(task,17)
            for t,frame in enumerate(frames):
                codes=vision.encode([frame]);history.append(codes)
                if t%10==0:program=old.generate(task,torch.stack(history[-11:],1),t)
                out=old.step(program,codes,memory,previous,t);memory=out['state_values'];previous=out['probabilities'].argmax(-1);int(previous[0])
        def predictive(encoder):
            found=0
            for t in [0,10,20]:
                begin=max(0,t-10);codes=encoder(frames[begin:t+1])[None]
                previous=torch.tensor([[int(sample[j-1]['label']) if j else 17 for j in range(begin,t+1)]],device='cuda')
                program=model.generate(task,codes,previous,torch.tensor([t-begin+1],device='cuda'),start=t)
                state=model.ground(program,codes[:,-1]).argmax(-1).cpu().numpy()[0]
                found+=plan(export_program(program),state,horizon=10)['found']
            return found
        old_s,_=duration(reactive);serial_s,found=duration(lambda:predictive(vision.encode));batch_s,found_batch=duration(lambda:predictive(batched))
        measurements.append(dict(reactive_seconds=old_s,predictive_serial_seconds=serial_s,predictive_batched_seconds=batch_s,
            generated_plans_found=found,batched_plans_found=found_batch))
result=dict(training_only=True,checkpoint=a.checkpoint,images=sum(map(len,images)),tokens=total,changed_tokens=changed,
    identical_programs=program_equal,program_count=programs,serial_vision_seconds=sum(serial),batched_vision_seconds=sum(parallel),
    vision_speedup=sum(serial)/sum(parallel),replay=measurements,
    replay_scope='21 fixed recorded images; neural computation and goal search only, not physical execution or navigation success',
    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
