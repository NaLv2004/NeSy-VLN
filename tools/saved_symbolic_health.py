"""Audit saved trajectories without loading a model or executing an environment.

Use real within-program state replacements in addition to artificial truth flips.
An action proof must reproduce its logged support, and forced expert actions are
explicitly separated from autonomous actions.
"""
import argparse,collections,json,math
from pathlib import Path
import numpy as np


def noisy_or(x,axis=0):
    return -np.expm1(np.log1p(-np.clip(x,0,1-1e-6)).sum(axis=axis))


def action_proof(program,belief,confidence):
    k=len(belief);n=len(program['weights'])//6;part=slice(2*n,4*n)
    literals=np.r_[1.,belief*confidence,(1-belief)*confidence]
    activation=program['weights'][part]*(program['premises'][part]@literals).prod(-1)
    contribution=activation[:,None]*program['action_heads'][part]
    support=noisy_or(contribution)
    return support/support.sum(),support,contribution


def mean(xs):return float(np.mean(xs)) if xs else None


def audit_trace(folder):
    t=json.loads((folder/'trajectory.json').read_text());records=t['steps']
    versions=collections.defaultdict(list)
    for r in records:versions[r['version']].append(r)
    errors=[];real_tv=[];flip_tv=[];real_changed=flip_changed=steps=0
    predicate_stds=[];variable_programs=0;expert_correct=expert_count=0
    expert_nll=[];constant_nll=[];teacher_error=[];camera_changes=[];belief_changes=[]
    greedy=[];program_stats=[]
    for version,rows in versions.items():
        raw=json.loads((folder/f'program_{version:03d}.json').read_text())['program']
        p={k:np.asarray(v,dtype=float) for k,v in raw.items() if k!='version'}
        b=np.asarray([r['belief'] for r in rows]);c=np.asarray([r['confidence'] for r in rows])
        observations=np.asarray([r['observed_truth'] for r in rows]);average=b.mean(0)
        if len(rows)>1:predicate_stds.append(float(observations.std(0).mean()))
        choices=[]
        for index,row in enumerate(rows):
            actual,support,contribution=action_proof(p,b[index],c[index])
            constant,_,_=action_proof(p,average,c[index])
            flipped,_,_=action_proof(p,1-b[index],c[index])
            errors.extend([float(np.max(np.abs(support-np.asarray(row['support'])))),
                           float(np.max(np.abs(contribution-np.asarray(row['contribution']))))])
            choices.append(int(actual.argmax()));greedy.append(int(actual.argmax()))
            real_changed+=int(actual.argmax()!=constant.argmax());flip_changed+=int(actual.argmax()!=flipped.argmax());steps+=1
            real_tv.append(float(np.abs(actual-constant).sum()/2));flip_tv.append(float(np.abs(actual-flipped).sum()/2))
            if row.get('expert',False):
                expert_count+=1;expert_correct+=int(actual.argmax()==row['action'])
                expert_nll.append(-math.log(max(float(actual[row['action']]),1e-12)))
                constant_nll.append(-math.log(max(float(constant[row['action']]),1e-12)))
            if 'prediction_error' in row:teacher_error.append(row['prediction_error'])
            if index:
                camera_changes.append(rows[index-1]['feedback'][1])
                belief_changes.append(float(np.abs(observations[index]-observations[index-1]).mean()))
        variable_programs+=int(len(set(choices))>1)
        d=p['definitions'];d=d/np.linalg.norm(d,axis=1,keepdims=True)
        gram=d@d.T;off=gram[~np.eye(len(d),dtype=bool)];n=len(p['weights'])//6
        heads=p['action_heads'][2*n:4*n].argmax(-1)
        program_stats.append({'version':int(version),'steps':len(rows),
            'distinct_real_greedy_actions':len(set(choices)),
            'mean_abs_definition_cosine':float(np.abs(off).mean()),
            'distinct_rule_head_argmax':len(set(heads.tolist())),
            'observed_truth_min':float(observations.min()),'observed_truth_max':float(observations.max()),
            'mean_observation_std':float(observations.std(0).mean())})
    metrics=t.get('metrics',{})
    return {'episode_id':t['episode_id'],'category':t['category'],'domain':t['domain'],
            'distance_group':t['distance_group'],'success':bool(metrics.get('success')),
            'steps':steps,'programs':len(versions),'real_state_variable_programs':variable_programs,
            'real_mean_replacement_action_changes':real_changed,'truth_flip_action_changes':flip_changed,
            'real_mean_replacement_tv':mean(real_tv),'truth_flip_tv':mean(flip_tv),
            'max_proof_error':max(errors,default=0.),'mean_within_program_observation_std':mean(predicate_stds),
            'expert_steps':expert_count,'expert_correct':expert_correct,'expert_nll':mean(expert_nll),
            'constant_state_expert_nll':mean(constant_nll),'mean_prediction_error':mean(teacher_error),
            'collisions':sum(bool(r['collision']) for r in records),
            'actions':[r['action'] for r in records],'greedy_actions':greedy,
            'final_distance':metrics.get('distance'),'final_visible_pixels':metrics.get('visible_pixels'),
            'program_details':program_stats,
            'visual_changes':camera_changes,'predicate_changes':belief_changes}


def summarize(rows):
    steps=sum(r['steps'] for r in rows);programs=sum(r['programs'] for r in rows)
    expert=sum(r['expert_steps'] for r in rows)
    result={'episodes':len(rows),'successes':sum(r['success'] for r in rows),'steps':steps,
            'proof_max_error':max((r['max_proof_error'] for r in rows),default=0.),
            'programs':programs,'programs_with_real_greedy_changes':sum(r['real_state_variable_programs'] for r in rows),
            'real_state_mean_replacement_changes':sum(r['real_mean_replacement_action_changes'] for r in rows),
            'truth_flip_changes':sum(r['truth_flip_action_changes'] for r in rows),
            'mean_within_program_observation_std':mean([r['mean_within_program_observation_std'] for r in rows if r['mean_within_program_observation_std'] is not None]),
            'expert_steps':expert,'expert_correct':sum(r['expert_correct'] for r in rows),
            'expert_accuracy':sum(r['expert_correct'] for r in rows)/expert if expert else None,
            'collision_action_rate':sum(r['collisions'] for r in rows)/max(1,steps)}
    for metric in ['real_mean_replacement_tv','truth_flip_tv','mean_prediction_error']:
        result[metric]=sum(r[metric]*r['steps'] for r in rows if r[metric] is not None)/max(1,steps)
    for key in ['domain','category','distance_group']:
        groups=collections.defaultdict(list)
        for r in rows:groups[r[key]].append(r)
        result[key]={k:{'n':len(v),'successes':sum(r['success'] for r in v)} for k,v in groups.items()}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--update',type=int)
    p.add_argument('--suite',choices=['training','development','fixed80','demonstration'],default='training')
    args=p.parse_args();run=Path(args.run);cfg=json.loads((run/'config.json').read_text())
    if cfg.get('discrete_reasoning',False):
        raise RuntimeError('Discrete neural rules require audit_discrete_trace.py and their saved rule-network snapshot')
    if args.suite=='training':
        root=run/'traces'
        candidates=sorted(root.glob('update_*'))
        architecture=json.loads((run/'architecture.json').read_text())
        expected=cfg['rollouts_per_rank']*architecture['world_size']
        complete=[f for f in candidates if len(list(f.glob('*/trajectory.json')))==expected]
        folder=root/f'update_{args.update:06d}' if args.update is not None else complete[-1]
        paths=sorted(folder.glob('*/trajectory.json'))
        if len(paths)!=expected:raise RuntimeError('Training trace snapshot is incomplete')
    elif args.suite=='demonstration':
        root=run/'demonstration_traces'
        folder=root/f'update_{args.update:06d}' if args.update is not None else sorted(root.glob('update_*'))[-1]
        paths=[folder/'trajectory.json']
    else:
        root=run/'evaluations'
        complete=sorted(f for f in root.glob('step_*') if (f/args.suite/'summary.json').exists())
        step=root/f'step_{args.update:06d}' if args.update is not None else complete[-1]
        folder=step/args.suite
        if not (folder/'summary.json').exists():raise RuntimeError('Evaluation has not completed')
        paths=sorted(folder.glob('*/trajectory.json'))
    rows=[audit_trace(p.parent) for p in paths]
    result={'source':str(folder),'suite':args.suite,'summary':summarize(rows),'episodes':rows}
    out=run/'symbolic_health';out.mkdir(exist_ok=True)
    name=(folder.parent.name+'_'+folder.name if args.suite in ['development','fixed80'] else folder.name+'_'+args.suite)+'.json'
    (out/name).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'path':str(out/name),'summary':result['summary']},indent=2))


if __name__=='__main__':main()
