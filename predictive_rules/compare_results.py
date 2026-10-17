"""Report parity only for complete, audited evaluations of the same task IDs."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare(reference, candidate):
    reference,candidate=Path(reference),Path(candidate)
    old,new=read(reference/'summary.json'),read(candidate/'summary.json')
    old_manifest,new_manifest=read(reference/'manifest.json'),read(candidate/'manifest.json')
    if old_manifest['episode_ids']!=new_manifest['episode_ids']:
        raise ValueError('Evaluations use different task IDs or order')
    if old_manifest['data']!=new_manifest['data']:
        raise ValueError('Dataset provenance differs')
    if old_manifest['horizon']!=32 or new_manifest['horizon']!=32:
        raise ValueError('Evaluation action budgets differ')
    if old['episodes']!=80 or new['episodes']!=80:
        raise ValueError('A partial result cannot establish parity')
    audit=read(candidate/'independent_rule_audit.json')
    assert audit['all_verified'] and audit['episodes']==80
    assert new['boundary_gate_violations']==0
    assert not new_manifest['previous_reactive_model']
    assert new_manifest['revision_interval']==10
    counts={}
    for key,baseline in old['by_group'].items():
        result=new['by_group'][key]
        assert baseline['episodes']==result['episodes']
        counts[key]={'baseline':baseline['successes'],'candidate':result['successes'],
                     'tasks':baseline['episodes'],'delta':result['successes']-baseline['successes']}
    meets=all(counts[key]['delta']>=0 for key in ['all','normal_near_medium'])
    return dict(suite=new['suite'],success_parity_met=meets,comparisons=counts,
        all_transitions_verified=True,no_intermediate_neural_inference=True,
        baseline_collision_rate=old['collision_action_rate'],candidate_collision_rate=new['collision_action_rate'],
        note='Success count parity on these fixed tasks; not a claim of universal statistical equivalence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');a=p.parse_args()
    print(json.dumps(compare(a.reference,a.candidate),indent=2))
