"""Training-only, post-execution comparison of frozen-label world models.

Observed states in this file are diagnostics, never feedback to a controller.
Each checkpoint is evaluated against the same committed executed action prefix.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel, Config
from predictive_nesy.data import sha
from predictive_nesy.graph_subgoal_interpreter import Program


def audit(checkpoint, folder, count):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    commit = json.loads(Path(checkpoint).with_suffix('.COMMITTED.json').read_text())
    assert sha(checkpoint) == commit['sha256']
    model_class=VisualMemorySubgoalModel
    if saved['manifest']['architecture']=='anonymous_definition_conditioned_world_v12':
        from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld
        model_class=DefinitionConditionedWorld
    model = model_class(Config(**saved['model_config'])).eval()
    model.load_state_dict(saved['model'])
    collection_manifest=json.loads((folder/'manifest.json').read_text())
    own_programs=collection_manifest.get('policy_checkpoint',{}).get('sha256')==commit['sha256']
    rows = []
    with torch.no_grad():
        for file in sorted(folder.glob('*.npz'))[:count]:
            trace = json.loads((folder/'proofs'/file.stem/'trace.json').read_text())
            event = trace['programs'][0]
            assert event['start'] == 0
            with np.load(file, allow_pickle=False) as data:
                n = min(len(event['plan']['actions']), len(data['actions']), 10)
                codes = torch.from_numpy(np.concatenate([data['codes'][:n], data['actual'][n-1:n]], 0).astype(np.int64))[None]
                branches = torch.from_numpy(data['successors'][:n].astype(np.int64))
                task = torch.tensor([int(data['task'])])
                actions = data['actions'][:n].tolist()
            p = model.generate(task, codes[:, :1], torch.tensor([[17]]), torch.tensor([1]), start=0)
            labels = model.encode_sequence(p, codes)[0].argmax(-1)[0].tolist()
            table = p['transitions'].argmax(-1)[0]
            predicted = [labels[0]]
            for action in actions:
                predicted.append(int(table[predicted[-1], action]))
            teacher = [int(table[labels[t], action]) for t, action in enumerate(actions)]
            counts = defaultdict(Counter)
            branch_correct = 0
            stop_changes = 0
            total_branches = 0
            for t in range(n):
                _, memory, previous, old_codes = model.encode_sequence(p, codes[:, :t+1])
                feature = model.observation(model.embed(branches[t]))
                memory, feature, _ = model.advance(memory.expand(17, -1), previous.expand(17, -1),
                    old_codes.expand(17, -1), feature, branches[t])
                features = F.normalize(feature+model.memory_projection(memory), dim=-1)
                definitions = F.normalize(p['definitions'][0], dim=-1)
                successors = (features @ definitions.T).argmax(-1).tolist()
                assert successors[actions[t]] == labels[t+1], 'Executed/probed target-label mismatch'
                for a, successor in enumerate(successors):
                    counts[labels[t], a][successor] += 1
                    branch_correct += int(table[labels[t], a]) == successor
                stop_changes += successors[0] != labels[t]
                total_branches += 17
            rows.append(dict(episode=file.stem,actions=actions,labels=labels,predicted=predicted,
                source_checkpoint_initial=int(event['plan']['states'][0][0]),
                initial_matches_source=labels[0] == int(event['plan']['states'][0][0]),
                autonomous_correct=sum(x == y for x,y in zip(predicted[1:], labels[1:])),
                one_step_correct=sum(x == y for x,y in zip(teacher, labels[1:])),
                first_correct=predicted[1] == labels[1],steps=n,
                branch_correct=branch_correct,branches=total_branches,
                best_deterministic_branch_correct=sum(max(c.values()) for c in counts.values()),
                conflicting_rows=sum(len(c)>1 for c in counts.values()),
                branch_rule_rows=len(counts),stop_changes_in_observed_labels=stop_changes,
                inferred_stop_rule_changes=sum(int(table[s,0])!=s for s in labels[:-1])))
            if own_programs:
                executable=Program.load(folder/'proofs'/file.stem/event['file'])
                actual_program={**p,'definitions':torch.tensor(executable.definitions.copy())[None]}
                actual_labels=model.encode_sequence(actual_program,codes)[0].argmax(-1)[0].tolist()
                executed=[int(x['after'][0]) for x in trace['steps'][:n]]
                runtime_rules=executable.transitions.argmax(-1)
                assert actual_labels[0]==event['initial_state'][0]
                rows[-1].update(executed_rule_autonomous_correct=sum(a==b for a,b in zip(executed,actual_labels[1:])),
                    executed_rule_one_step_correct=sum(int(runtime_rules[actual_labels[t],a])==actual_labels[t+1] for t,a in enumerate(actions)),
                    executed_rule_first_correct=executed[0]==actual_labels[1],
                    runtime_vs_regenerated_rule_rows_changed=int((runtime_rules!=table.numpy()).sum()),
                    runtime_vs_regenerated_observed_labels_changed=sum(a!=b for a,b in zip(actual_labels,labels)))
    keys=['steps','autonomous_correct','one_step_correct','first_correct','branch_correct','branches',
          'best_deterministic_branch_correct','conflicting_rows','branch_rule_rows',
          'stop_changes_in_observed_labels','inferred_stop_rule_changes']
    if own_programs:keys+=['executed_rule_autonomous_correct','executed_rule_one_step_correct','executed_rule_first_correct',
        'runtime_vs_regenerated_rule_rows_changed','runtime_vs_regenerated_observed_labels_changed']
    return dict(checkpoint=str(checkpoint),sha256=commit['sha256'],update=saved['update'],samples=len(rows),
        matching_collection_checkpoint=own_programs,
        initial_matches_source=sum(r['initial_matches_source'] for r in rows),
        **{k:sum(r[k] for r in rows) for k in keys},rows=rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--folder',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=48)
    a=p.parse_args();torch.set_num_threads(2)
    result=dict(training_only=True,post_execution_only=True,teacher_forced_metrics_never_used_at_runtime=True,
        results=[audit(Path(c),Path(a.folder),a.count) for c in a.checkpoint])
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({**result,'results':[{k:v for k,v in r.items() if k!='rows'} for r in result['results']]}),flush=True)
