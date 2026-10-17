"""Reuse the audited training-only labels; compare anonymous vocabulary sizes."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
s=(root/'audit_transition_identifiability_v23.py').read_text(encoding='utf-8')
def change(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
change("output=root/'diagnostics/transition_identifiability_v23.json';assert not output.exists()",
    "output=root/'diagnostics/state_resolution_v24.json';assert not output.exists()")
change("rules=defaultdict(Counter);programs=defaultdict(Counter);stats={};began=time.time()",
    "rules=defaultdict(Counter);programs=defaultdict(Counter);stats={};began=time.time()\n    latent_rows=[];program_keys=[];layout_ids=[]")
change("after=torch.einsum('bd,bkd->bk',nf,defs).reshape(len(selected),17,128).argmax(-1)",
    "after=torch.einsum('bd,bkd->bk',nf,defs).reshape(len(selected),17,128).argmax(-1)\n                current_latent=F.normalize(previous+model.memory_projection(memory),dim=-1)\n                latent_rows.append(torch.cat([current_latent[:,None],nf.reshape(len(selected),17,256)],1))\n                program_keys.extend(keys)\n                layout_ids.extend(int(name.split('_')[2]) for name,t in selected)")
change("output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['initial_observed_branches','conflicts']}),flush=True)",
    "from state_resolution_probe_v24 import probe\n    probe(torch.cat(latent_rows),program_keys,layout_ids,result,output)")
(root/'audit_state_resolution_v24.py').write_text(s,encoding='utf-8')
print('Prepared an offline anonymous-state resolution probe; no navigation rollout or optimizer update.')
