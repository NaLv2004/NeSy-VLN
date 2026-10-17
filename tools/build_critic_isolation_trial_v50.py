"""Derive a matched trial from the already audited PPO implementation."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'scripts/train_token_revision_rl_v45.py').read_text()
edits={
 "parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=40)":
 "parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=40)\nparser.add_argument('--arm',choices=['shared_critic','detached_critic'],required=True)",
 "out=base/'token_revision_rl_v45';out.mkdir(exist_ok=True)":
 "out=base/'critic_isolation_rl_v50'/args.arm;out.mkdir(parents=True,exist_ok=True)",
 "source=base/'spatial_automaton_v43';source_commit=read(source/'COMMITTED')":
 "source=base/'recovery_rule_learning_v49/without_previous_action';source_commit=read(source/'COMMITTED')\nvocabulary_source=base/'spatial_automaton_v43'",
 "sha(source/'visual_vocabulary.pt')": "sha(vocabulary_source/'visual_vocabulary.pt')",
 "reviser=TokenProgramReviser(schema,len(tasks),tree_count)":
 "reviser=TokenProgramReviser(schema,len(tasks),tree_count,detach_value_features=args.arm=='detached_critic')",
 "vocabulary=torch.load(source/'visual_vocabulary.pt'": "vocabulary=torch.load(vocabulary_source/'visual_vocabulary.pt'",
 "manifest={'source_commit':source_commit,'data':bank.provenance":
 "manifest={'source_commit':source_commit,'arm':args.arm,'detach_value_features':args.arm=='detached_critic','previous_action_hidden':True,'data':bank.provenance",
 "torch.tensor([record['last_action']])": "torch.tensor([64])",
 "Learned integer visual tokens and past action tokens. Program weights frozen for ten executed actions.":
 "Learned integer visual token history. Prior-action slot always unknown64, matching source rules. Program weights frozen for ten actions.",
}
for old,new in edits.items():
 assert old in text,old
 text=text.replace(old,new)
text=text.replace('"""On-policy learning of ten-action symbolic programs over learned token facts."""',
 '"""Matched critic-gradient isolation on the corrected automatically learned rules."""')
(root/'scripts/train_critic_isolation_rl_v50.py').write_text(text)
print('Created audited-source derivative train_critic_isolation_rl_v50.py')
