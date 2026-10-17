"""Replay saved soft inference numerically and intervene on beliefs/rules, without a simulator."""
import json,collections
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parents[1]/'output/inference_diagnosis_step30'
read=lambda p:json.loads(p.read_text())
rng=np.random.default_rng(20261010)

def noisy_or(x,axis=0):
    return -np.expm1(np.log1p(-np.clip(x,0,1-1e-6)).sum(axis=axis))

def infer(p,b,c,states=True,disable=None):
    b=np.asarray(b).copy();c=np.asarray(c).copy()
    for start,end in ([(0,8),(8,16)] if states else []):
        literal=np.r_[1,b*c,(1-b)*c]
        strength=p['weights'][start:end]*(p['premises'][start:end]@literal).prod(-1)
        added=noisy_or(strength[:,None]*p['state_heads'][start:end])
        positive=1-(1-b*c)*(1-added);c=1-(1-c)*(1-added)
        b=np.clip(positive/np.maximum(c,1e-6),0,1)
    literal=np.r_[1,b*c,(1-b)*c]
    strength=p['weights'][16:32]*(p['premises'][16:32]@literal).prod(-1)
    if disable is not None:strength[disable]=0
    contrib=strength[:,None]*p['action_heads'][16:32]
    support=noisy_or(contrib);probs=support/support.sum()
    return probs,contrib,strength

def recover_input(row):
    b=np.array(row['belief']);c=np.array(row['confidence'])
    for layer in reversed(row['state_inference']):
        added=noisy_or(np.array(layer['contribution']))
        old_c=(c-added)/(1-added)
        positive=(b*c-added)/(1-added)
        b=positive/old_c;c=old_c
    return np.clip(b,0,1),np.clip(c,0,1)

groups=collections.defaultdict(list);cases=[];errors=[];all_steps=[]
for folder in sorted((base/'fixed80').iterdir()):
    t=read(folder/'trajectory.json');programs={}
    for f in folder.glob('program_*.json'):
        q=read(f)['program'];p={k:np.asarray(v) for k,v in q.items() if k!='version'}
        version=q['version'];programs[version]=p;kind='initial' if version==0 else 'revision'
        ah=p['action_heads'][16:32];defs=p['definitions'];dn=defs/np.linalg.norm(defs,axis=1,keepdims=True)
        cosine=dn@dn.T;lit=p['premises'][16:32,0,1:]
        heads=ah.argmax(-1);majority=int(np.bincount(heads,minlength=17).argmax())
        probes=[np.zeros(16),np.ones(16),np.full(16,.5),*[rng.uniform(0,1,16) for _ in range(100)]]
        # Artificial full-range belief/confidence perturbations; no images are changed.
        outcomes=[];distributions=[]
        for i,b in enumerate(probes):
            c=np.full(16,.7) if i<3 else rng.uniform(.05,1,16)
            pr,_,_=infer(p,b,c);outcomes.append(int(pr.argmax()));distributions.append(pr)
        stat={'episode_id':t['episode_id'],'version':version,'rule_head_counts':dict(collections.Counter(map(int,heads))),
              'majority_action':majority,'all_rules_same_action':bool((heads==heads[0]).all()),
              'mean_head':ah.mean(0).tolist(),'head_across_rules_std':ah.std(0).mean().item(),
              'mean_definition_cosine':float(cosine[~np.eye(16,dtype=bool)].mean()),
              'min_definition_cosine':float(cosine[~np.eye(16,dtype=bool)].min()),
              'premise_first_max':float(lit.max(-1).mean()),
              'premise_first_entropy':float(-(lit*np.log(lit.clip(1e-12))).sum(-1).mean()),
              'premise_second_constant':float(p['premises'][16:32,1,0].mean()),
              'predicate_positive_negative_mass_gap':float(np.abs(lit[:,:16].sum(-1)-lit[:,16:].sum(-1)).mean()),
              'counterfactual_action_counts':dict(collections.Counter(outcomes)),
              'counterfactual_max_probability_range':float(np.ptp(np.asarray(distributions),axis=0).max())}
        groups[kind].append(stat)
    step_rows=[]
    for i,row in enumerate(t['steps']):
        p=programs[row['version']];b,c=recover_input(row)
        prob,contrib,strength=infer(p,b,c)
        saved=np.asarray(row['support']);saved=saved/saved.sum()
        errors.append(float(np.abs(prob-saved).max()))
        errors.append(float(np.abs(contrib-np.asarray(row['contribution'])).max()))
        no_states=infer(p,b,c,states=False)[0]
        flipped=infer(p,1-b,c)[0]
        flat=infer(p,np.full(16,.5),c)[0]
        disabled=[int(infer(p,b,c,disable=r)[0].argmax()) for r in range(16)]
        action=int(prob.argmax());observed=np.asarray(row['observed_truth'])
        rec={'step':i+1,'version':row['version'],'action':action,'probability':prob.tolist(),
             'no_state_rules_action':int(no_states.argmax()),'flip_beliefs_action':int(flipped.argmax()),
             'flat_beliefs_action':int(flat.argmax()),'single_rule_deletion_changes':sum(a!=action for a in disabled),
             'no_state_rules_tv':float(np.abs(prob-no_states).sum()/2),
             'flipped_beliefs_tv':float(np.abs(prob-flipped).sum()/2),
             'flat_beliefs_tv':float(np.abs(prob-flat).sum()/2),
             'observed_mean':float(observed.mean()),'observed_std_across_predicates':float(observed.std()),
             'observed_min':float(observed.min()),'observed_max':float(observed.max()),
             'belief_mean':float(np.mean(row['belief'])),'confidence_mean':float(np.mean(row['confidence'])),
             'top_rules':[(16+int(r),float(strength[r]),float(contrib[r,action])) for r in np.argsort(contrib[:,action])[-3:][::-1]],
             'prediction_error':row['prediction_error']}
        step_rows.append(rec);all_steps.append(rec)
    cases.append({'episode_id':t['episode_id'],'success':bool(t['metrics'].get('success')),'metrics':t['metrics'],
                  'steps':step_rows})

def stats(xs):
    return {'mean':float(np.mean(xs)),'min':float(np.min(xs)),'max':float(np.max(xs))}
summary={'numerical_replay_max_abs_error':max(errors),'steps':len(all_steps),'programs':{},
         'interventions':{},'observed_predicate_std':stats([r['observed_std_across_predicates'] for r in all_steps])}
for kind,rows in groups.items():
    summary['programs'][kind]={'count':len(rows),'all_rules_same_action_count':sum(r['all_rules_same_action'] for r in rows),
        'majority_actions':dict(collections.Counter(r['majority_action'] for r in rows)),
        'counterfactual_total_action_counts':dict(sum([collections.Counter(r['counterfactual_action_counts']) for r in rows],collections.Counter()))}
    for key in ['mean_definition_cosine','min_definition_cosine','premise_first_max','premise_first_entropy',
                'premise_second_constant','head_across_rules_std','counterfactual_max_probability_range']:
        summary['programs'][kind][key]=stats([r[key] for r in rows])
    summary['programs'][kind]['mean_head']=np.array([r['mean_head'] for r in rows]).mean(0).tolist()
for name in ['no_state_rules','flip_beliefs','flat_beliefs']:
    summary['interventions'][name]={'changed_action':sum(r[name+'_action']!=r['action'] for r in all_steps),'total':len(all_steps)}
summary['interventions']['single_rule_deletion']={'changed_action':sum(r['single_rule_deletion_changes'] for r in all_steps),'total':16*len(all_steps)}
summary['interventions']['no_state_rules_probability_tv']=stats([r['no_state_rules_tv'] for r in all_steps])
summary['interventions']['flipped_beliefs_probability_tv']=stats([r['flipped_beliefs_tv'] for r in all_steps])
summary['interventions']['flat_beliefs_probability_tv']=stats([r['flat_beliefs_tv'] for r in all_steps])
result={'summary':summary,'programs':groups,'cases':cases}
(base/'logic_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
for episode in ['P25_validation_200_chair_medium_1_000','P25_validation_200_chair_far_3_000']:
    case=next(r for r in cases if r['episode_id']==episode)
    print('EXAMPLE',episode,case['success'],case['metrics']['distance'])
    for s in [case['steps'][0],case['steps'][9],case['steps'][10]]:
        print(json.dumps({k:s[k] for k in ['step','version','action','observed_mean','observed_std_across_predicates','belief_mean','confidence_mean','probability','top_rules']}))
