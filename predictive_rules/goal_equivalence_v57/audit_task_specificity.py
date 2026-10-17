"""Frozen CPU audit of different goals from exactly the same training RGB.

Geometry is used only to certify audit labels, never fed into the model. A
triangle inequality lower bound certifies that a successful endpoint for one
task cannot meet the other task's existing distance threshold. No map is built.
"""
import json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_definition_rules_v56'))
from model import FactoredWorld,Config,hard
from data import sha

@torch.inference_mode()
def main():
    torch.set_num_threads(2);began=time.time();out=ROOT/'diagnostic_goal_equivalence_v57'
    manifest=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
    p=Path(manifest['data']['source'])/'training_rows.json';assert sha(p)==manifest['data']['rows_sha256']
    rows=json.loads(p.read_text());raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    episodes=defaultdict(list)
    for i,r in enumerate(rows):episodes[r['episode_id']].append(i)
    reference=json.loads(Path('/home/user/xjy/strips_wm_navigation_20261009/configs/experiment.json').read_text())
    radius=json.loads(Path(reference['frozen_config']).read_text())['rewards']['success_distance_m']
    pairs=[]
    for p in sorted(Path('/home/user/xjy/nesy_vla_20261010/data/paired_goals_v12/pairs').glob('*/COMMITTED.json')):
        d=json.loads(p.read_text())
        if not d['both_successful'] or not d['same_initial_rgb']:continue
        assert d['scene'] in manifest['data']['train_layouts']
        branches=d['branches']
        if not all(b['episode']['episode_id'] in episodes for b in branches):continue
        assert len(branches)==2 and {b['episode']['category'] for b in branches}=={'chair','table'}
        va,vb=[np.asarray(b['episode']['viewpoints']) for b in branches]
        separation=float(np.linalg.norm(va[:,None]-vb[None],axis=-1).min())
        for own,other in ((branches[0],branches[1]),(branches[1],branches[0])):
            # d(other_terminal, own_goal_set) >= separation - d(other_terminal, other_goal_set).
            # The stored geodesic terminal distance upper-bounds its Euclidean distance.
            lower=separation-other['metrics']['distance']
            if lower<=radius+1e-4:continue
            pairs.append(dict(episode=own['episode']['episode_id'],other_episode=other['episode']['episode_id'],
                certified_wrong_goal_distance_lower_bound=lower,pair=p.parent.name))
    assert pairs,'No geometrically certified negative pair: do not make an unsupported label'
    def batch(examples,key):
        x=np.zeros((len(examples),34,49,128),np.float16);previous=np.full((len(examples),34),17,np.int64);end=[];task=[]
        for b,r in enumerate(examples):
            ids=episodes[r[key]];n=len(ids);assert n<=33 and rows[ids[-1]]['label']==0
            x[b,:n]=raw[ids];x[b,n]=x[b,n-1];previous[b,1:n+1]=[rows[i]['label'] for i in ids]
            end.append(n);task.append(['chair','table'].index(rows[ids[0]]['category']))
        return torch.tensor(x),torch.tensor(previous),torch.tensor(end),torch.tensor(task)
    results={}
    for variant in ('source','aligned'):
        if variant=='source':
            p=ROOT/'diagnostic_temporal_symbols_v55/source.pt';saved=torch.load(p,map_location='cpu',weights_only=False)
            model=FactoredWorld(Config(**saved['model_config']))
            head=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)
            assert sha(p)==head['source']['sha256'];model.load_initial(saved['model'],head['head']);del head
        else:
            saved=torch.load(out/'probe.pt',map_location='cpu',weights_only=False)
            model=FactoredWorld(Config(**saved['config']));model.load_state_dict(saved['model'])
        del saved;model.eval();observations=[]
        for start in range(0,len(pairs),16):
            examples=pairs[start:start+16];x,prev,end,task=batch(examples,'episode');other,oprev,oend,otask=batch(examples,'other_episode')
            assert torch.equal(x[:,0],other[:,0]) and torch.equal(1-task,otask)
            ix=torch.arange(len(x));context=model.context_sequence(x,hard(model.observe(x)),prev,task)[:,0]
            gm,go=model.goal_logits(context,task);mi,oi=gm.argmax(-1),go.argmax(-1)
            def terminal(x,p,e):
                logits,z,_=model.observation_encoding(x,context);obs=hard(logits)
                lm,_,_=model.memory_sequence(obs,p,z)
                pm,po=model.transition(context,hard(lm[ix,e-1]),obs[ix,e-1],torch.zeros(len(x),dtype=torch.long))
                return logits[ix,e].argmax(-1),lm[ix,e].argmax(-1),pm.argmax(-1),po.argmax(-1)
            own_o,own_m,pm,po=terminal(x,prev,end);wrong_o,wrong_m,wpm,wpo=terminal(other,oprev,oend)
            for i,r in enumerate(examples):
                observations.append(dict(**r,
                    correct_endpoint_is_goal=bool((own_o[i]==oi[i])&(own_m[i]==mi[i])),
                    wrong_endpoint_observation_is_goal=bool(wrong_o[i]==oi[i]),
                    wrong_endpoint_joint_is_goal=bool((wrong_o[i]==oi[i])&(wrong_m[i]==mi[i])),
                    wrong_endpoint_stop_reaches_goal=bool((wpm[i]==mi[i])&(wpo[i]==oi[i])),
                    own_and_wrong_same_observation=bool(own_o[i]==wrong_o[i])))
        fields=('correct_endpoint_is_goal','wrong_endpoint_observation_is_goal','wrong_endpoint_joint_is_goal','wrong_endpoint_stop_reaches_goal','own_and_wrong_same_observation')
        results[variant]=dict(episodes=len(observations),metrics={k:sum(r[k] for r in observations)/len(observations) for k in fields},rows=observations)
    result=dict(time=time.time(),seconds=time.time()-began,training_only=True,original_dev80_used=False,finalfixed80_used=False,
        existing_success_radius_m=radius,geometry_input_to_model=False,new_rollouts=0,results=results,
        caveat='Training paired-goal data only; predicts symbolic terminal discrimination, not closed-loop navigation.')
    (out/'task_specificity_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:dict(episodes=v['episodes'],metrics=v['metrics']) for k,v in results.items()}),flush=True)
if __name__=='__main__':main()
