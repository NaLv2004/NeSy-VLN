import hashlib,json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');code=base/'lossless_dataset_v7_code';out=base/'data/lossless_demonstrations_v7'
out.mkdir(parents=True,exist_ok=True)
provenance=json.loads((base/'symbolic_loss_v6/runs/loss_reference/data_provenance.json').read_text())
source=out/'source.json'
if source.exists():assert json.loads(source.read_text())==provenance
else:source.write_text(json.dumps(provenance,indent=2))
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python';jobs={};records=[]

def persist(state):
    temporary=out/'controller.tmp';temporary.write_text(json.dumps({'time':time.time(),'pid':os.getpid(),'state':state,'jobs':records},indent=2))
    temporary.replace(out/'controller.json')

for index,gpu in enumerate([0,3,6]):
    if (out/f'shard_{index}.json').exists():continue
    command=[python,'-u',str(code/'build_lossless_demonstrations.py'),'--shard',str(index),
        '--shards','3','--gpu',str(gpu),'--output',str(out)]
    with (out/f'worker_{index}.log').open('ab') as log:
        process=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs[index]=process;records.append({'shard':index,'gpu':gpu,'pid':process.pid,'command':command});persist('generating')
for index,process in jobs.items():
    code=process.wait()
    if code!=0:
        persist('worker_failed');raise RuntimeError(f'Worker {index} failed with {code}; resume only that shard from committed episodes')
episodes=[]
for index in range(3):
    part=json.loads((out/f'shard_{index}.json').read_text());assert part['source_hash']==provenance['demonstration_hash']
    episodes.extend(part['episodes'])
assert len(episodes)==provenance['demonstrations'] and len({r['episode_id'] for r in episodes})==len(episodes)
examples=[]
for episode in sorted(episodes,key=lambda r:r['episode_id']):
    assert episode['scene'] in provenance['train_layouts']
    if not episode['success']:continue
    frames=episode['frames'];assert frames[-1]['action']==0
    for index,frame in enumerate(frames):
        row={k:v for k,v in frame.items() if k!='sha256'}
        row['context']=[frames[max(0,index-2+j)]['image'] for j in range(3)]
        examples.append(row)
temporary=out/'examples.tmp';temporary.write_text(json.dumps(examples,separators=(',',':')));temporary.replace(out/'examples.json')
summary={'time':time.time(),'examples':len(examples),'source_episodes':len(episodes),
    'successful_episodes':sum(r['success'] for r in episodes),'failed_episodes':[r['episode_id'] for r in episodes if not r['success']],
    'max_canonical_source_pixel_mae':max(r['max_recorded_pixel_mae'] for r in episodes),
    'examples_sha256':hashlib.sha256((out/'examples.json').read_bytes()).hexdigest(),
    'images':'Lossless raw online RGB. No upsample or JPEG transport.',
    'feedback':'Actual public collision feedback, with no private geometry or reward labels passed to the policy.',
    'data_provenance':provenance}
(out/'READY.json').write_text(json.dumps(summary,indent=2));persist('complete');print(json.dumps(summary,indent=2))
