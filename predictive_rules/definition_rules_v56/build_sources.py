"""Build an isolated v56 experiment; never edit the live v55 source."""
from pathlib import Path
import ast
import shutil

here=Path(__file__).resolve().parent
previous=here.parent/'temporal_symbols_v55'
for name in ('data.py','objectives.py','temporal_objective.py','evaluate.py','planner.py','storage.py','program_archive.py'):
    shutil.copy2(previous/name,here/name)
storage=(here/'storage.py').read_text().replace('run_temporal_symbols_v55','run_definition_rules_v56')
(here/'storage.py').write_text(storage)
objective=(here/'objectives.py').read_text()
assert 'sm,ob=hard(next_m),hard(next_o)' in objective
objective=objective.replace('sm,ob=hard(next_m),hard(next_o)',
    '# Future rows are still selected by autonomous integer predictions.\n        # Stop surrogate gradients through those discrete row addresses;\n        # every selected row retains its own measured successor supervision.\n        sm,ob=hard(next_m).detach(),hard(next_o).detach()')
(here/'objectives.py').write_text(objective)
shutil.copy2(previous/'model.py',here/'base_model.py')
base=(previous/'model.py').read_text()
extra=(here/'model_extension.py').read_text().replace('from base_model import FactoredWorld as BaseWorld, Config, hard\n','').replace('from definition_rule import DefinitionRule, definition_inputs\n','')
(here/'model.py').write_text(base+'\n\nBaseWorld=FactoredWorld\n\n'+(here/'definition_rule.py').read_text()+'\n\n'+extra)
text=(previous/'train.py').read_text()
text=text.replace('551017','561017').replace('temporal_symbols_v55','definition_rules_v56')
text=text.replace("model.load_v54(previous['model'])", "pilot=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)\n        assert pilot['source']['sha256']==source_sha\n        model.load_initial(previous['model'],pilot['head'])\n        del pilot")
text=text.replace('if n.startswith(representation_prefixes)', 'if p.requires_grad and n.startswith(representation_prefixes)')
text=text.replace('if not n.startswith(representation_prefixes)', 'if p.requires_grad and not n.startswith(representation_prefixes)')
text=text.replace('Best compatible v52/v54 weights plus new training-only temporal discriminator; fresh AdamW; unchanged representation/rule rates',
    'Pinned v52 update2000 plus CPU-trained definition-aware categorical row generator; fresh AdamW, unchanged representation/rule learning rates')
text=text.replace('Best compatible v52/v54; add density-ratio training of hard symbolic states; all runtime rule/planner behavior preserved before optimization',
    'Pinned v52 update2000; new shared categorical scorer consumes current and candidate generated observation definitions. Runtime remains exported hard-state rules.')
text=text.replace("runtime_action_selection='NumPy exhaustive probability derivation with normalized float64 dual success/failure arithmetic v50')", "runtime_action_selection='NumPy exhaustive probability derivation with normalized float64 dual success/failure arithmetic v50',\n        definition_aware_transition=True,observation_rename_equivariant=True,\n        no_continuous_transition_carrier=True,\n        pilot_head_sha256=sha(ROOT/'diagnostic_definition_rules_v56/head.pt'),\n        pilot_is_not_navigation_evidence=True)")
text=text.replace('definition_aware_transition=True,',
    "categorical_rollout_addresses_detached=True,definition_aware_transition=True,")
(here/'train.py').write_text(text)
text=(previous/'supervisor.py').read_text().replace('temporal_symbols_v55','definition_rules_v56').replace('29755','29756')
(here/'supervisor.py').write_text(text)
text=(previous/'inspect_run.py').read_text().replace('temporal_symbols_v55','definition_rules_v56')
text=text.replace("'temporal_ratio_loss','temporal_ratio_accuracy'", "'branch_observation_accuracy','hybrid_branch_observation_accuracy'")
(here/'inspect_run.py').write_text(text)
for p in here.glob('*.py'):ast.parse(p.read_text())
print('Built isolated v56 sources.')
