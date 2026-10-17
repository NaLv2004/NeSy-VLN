"""Measure rule error versus conflicting outcomes under the EXACT same program.

Training-only recorded probes. No environment queries, state labels, map, or
changes to the policy. All successor IDs use the source program's definitions.
"""
import collections, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_onpolicy_refresh_v60_training'))
from model import FactoredWorld,Config,hard
from data import sha
DATA=ROOT/'onpolicy_successors_v60';OUT=ROOT/'diagnostic_ensemble_rules_v61'

def summarize(groups):
    total=sum(sum(c.values()) for c in groups.values())
    repeats=[c for c in groups.values() if sum(c.values())>1]
    repeated_total=sum(sum(c.values()) for c in repeats)
    contradictions=[c for c in groups.values() if len(c)>1]
    # Leave one position's outcome out of each exact categorical premise.
    loo_correct=0
    for counts in repeats:
        for target,count in counts.items():
            other=counts.copy();other[target]-=1
            prediction=max(other,key=lambda k:(other[k],-k))
            loo_correct+=count*(prediction==target)
    return dict(outcomes=total,distinct_premises=len(groups),
        repeated_outcomes=repeated_total,contradictory_premises=len(contradictions),
        outcomes_in_contradictory_premises=sum(sum(c.values()) for c in contradictions),
        empirical_accuracy_ceiling=sum(max(c.values()) for c in groups.values())/total,
        repeated_premise_accuracy_ceiling=sum(max(c.values()) for c in repeats)/max(1,repeated_total),
        conditional_entropy_bits=sum(-v*math.log2(v/sum(c.values())) for c in groups.values() for v in c.values())/total,
        leave_one_position_out_coverage=repeated_total/total,
        leave_one_position_out_accuracy_on_covered=loo_correct/max(1,repeated_total))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);os.nice(10)
    assert not (OUT/'premise_audit.json').exists()
    started=time.time();commit=json.loads((DATA/'source.COMMITTED.json').read_text())
    assert sha(DATA/'source.pt')==commit['sha256']
    saved=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True)
    manifest=json.loads((DATA/'manifest.json').read_text());assert manifest['training_only']
    groups={name:collections.defaultdict(collections.Counter) for name in ('program_M_O_A','program_O_A')}
    correct=changed_correct=changed_count=total=observed_matches=positions=0
    episodes=0;programs=0;examples=[]
    with torch.inference_mode():
        for path in sorted((DATA/'counterfactual_onpolicy').glob('*.json')):
            r=json.loads(path.read_text());assert r['scene'] in manifest['train_layouts'] and r['training_only']
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:
                x=torch.tensor(data['features'].copy())[None]
                future=torch.tensor(data['successor_features'].copy())
                actions=torch.tensor(data['recorded_actions'].copy()).long()
            n=x.shape[1];previous=torch.cat([torch.tensor([17]),actions])[:n][None]
            task=torch.tensor([['chair','table'].index(r['category'])])
            contexts=model.context_sequence(x,hard(model.observe(x)),previous,task)
            z=F.normalize(model.visual(x.float().flatten(-2)),dim=-1)
            future_z=F.normalize(model.visual(future.float().flatten(-2)),dim=-1)
            for j,program in enumerate(r['programs']):
                begin=program['start'];end=r['programs'][j+1]['start'] if j+1<len(r['programs']) else n
                ctx=contexts[:,begin]
                definitions=F.normalize(model.observation_definitions[None]+model.observation_definition_delta(ctx),dim=-1)[0]
                target=(future_z[begin:end]@definitions.T).argmax(-1)
                current=(z[0,begin:end]@definitions.T).argmax(-1)
                steps=[r['steps'][t] if t<len(r['steps']) else r['failed_proof'] for t in range(begin,end)]
                m=torch.tensor([s['memory'] for s in steps]);o=torch.tensor([s['observed'] for s in steps])
                observed_matches+=int((current==o).sum());positions+=len(o)
                a=torch.arange(17)[None].expand(len(o),-1)
                pm,po=model.transition(ctx,
                    F.one_hot(m,32)[None,:,None].expand(1,-1,17,-1).float(),
                    F.one_hot(o,64)[None,:,None].expand(1,-1,17,-1).float(),a[None])
                pred=po[0].argmax(-1);changed=target!=o[:,None]
                total+=target.numel();correct+=int((pred==target).sum())
                changed_count+=int(changed.sum());changed_correct+=int(((pred==target)&changed).sum())
                for t in range(len(o)):
                    for action in range(17):
                        common=(r['episode'],j)
                        groups['program_M_O_A'][common+(int(m[t]),int(o[t]),action)][int(target[t,action])]+=1
                        groups['program_O_A'][common+(int(o[t]),action)][int(target[t,action])]+=1
                programs+=1
            episodes+=1
    for key,counts in groups['program_M_O_A'].items():
        if len(counts)>1 and len(examples)<12:
            examples.append(dict(episode=key[0],program=key[1],memory=key[2],observation=key[3],
                action=key[4],outcomes=dict(counts)))
    result=dict(time=time.time(),seconds=time.time()-started,source=commit,training_only=True,
        official_dev80_used=False,finalfixed80_used=False,optimizer_updates=0,episodes=episodes,
        programs=programs,positions=positions,observed_id_reproduction=observed_matches/positions,
        actual_source_accuracy=correct/total,actual_changed_accuracy=changed_correct/changed_count,
        groups={name:summarize(g) for name,g in groups.items()},examples=examples,
        caution='Empirical duplicate-premise bounds are optimistic on a finite recorded sample. Context-specific rule keys use identical generated definitions, not IDs compared across programs. Leave-one-position-out frequencies are diagnostic and never enter runtime planning.')
    (OUT/'premise_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
