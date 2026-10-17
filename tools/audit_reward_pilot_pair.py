import hashlib,json,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_reward_v11')
a=root/'runs/original';b=root/'runs/bounded_repeat_stop40'
def read(path):return json.loads(path.read_text())
left=json.loads((a/'rank0.jsonl').read_text().splitlines()[0]);right=json.loads((b/'rank0.jsonl').read_text().splitlines()[0])
assert left['update']==right['update']==1 and left['episode_ids']==right['episode_ids']
traces_a=list((a/'traces/update_000001').glob('*/trajectory.json'))
traces_b=list((b/'traces/update_000001').glob('*/trajectory.json'))
assert len(traces_a)==len(traces_b)==1
ta=read(traces_a[0]);tb=read(traces_b[0]);assert ta['episode_id']==tb['episode_id']
assert len(ta['steps'])==len(tb['steps'])
changed=[]
for i,(sa,sb) in enumerate(zip(ta['steps'],tb['steps'])):
    assert sa['action']==sb['action'] and sa['symbol_codes']==sb['symbol_codes']
    assert hashlib.sha256((traces_a[0].parent/sa['image']).read_bytes()).digest()==hashlib.sha256((traces_b[0].parent/sb['image']).read_bytes()).digest()
    assert abs(sa['reward']-sb['training_reward_audit']['original_reward'])<1e-7
    if abs(sa['reward']-sb['reward'])>1e-7:changed.append({'step':i,'action':sa['action'],'original':sa['reward'],'changed':sb['reward'],
        'training_terms':sb['training_reward_audit']['training_terms']})
result={'time':time.time(),'same_source_checkpoint':read(root/'initialization_source/COMMITTED'),
    'all8_training_episode_ids_match':True,'saved_trace':ta['episode_id'],'saved_trace_steps':len(ta['steps']),
    'saved_trace_images_actions_and_discrete_states_identical':True,'only_training_rewards_changed_in_saved_trace':True,
    'changed_reward_steps':changed,'success_criterion_unchanged':ta['metrics']['success']==tb['metrics']['success'],
    'note':'This audits identical initial rollout evidence, not later policy improvement.'}
(root/'first_update_pair_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
