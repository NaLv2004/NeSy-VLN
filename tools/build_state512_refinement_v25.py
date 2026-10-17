from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
s=(root/'initialize_state512_v25.py').read_text(encoding='utf-8')
def change(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
change("out=root/'initialization_state512_v25'","out=root/'initialization_state512_v25r1'")
change('    del old,state',"    candidate=root/'initialization_state512_v25/unaccepted_candidate.pt'\n    assert sha(candidate)==json.loads(candidate.with_suffix('.COMMITTED.json').read_text())['sha256']\n    resumed=torch.load(candidate,map_location='cpu',weights_only=False)\n    model.load_state_dict(resumed['model'])\n    del old,state,resumed")
change('lr=1e-3,weight_decay=1e-4','lr=5e-5,weight_decay=0.,eps=1e-10')
change('lr=1e-3 if step<=600 else 1e-4 if step<=3000 else 2e-5','lr=5e-5 if step<=3000 else 1e-5')
change('loss=F.mse_loss(predicted,centers[None].expand_as(predicted))','loss=256*F.mse_loss(predicted,centers[None].expand_as(predicted))')
change('mse=float(loss.detach())','mse=float(loss.detach())/256')
change('definitions_distillation_steps=12000','definitions_distillation_steps=24000,\n            refinement_initialization_sha256=sha(candidate),refinement_loss_scaling=256,refinement_weight_decay=0.')
(root/'refine_state512_v25.py').write_text(s,encoding='utf-8')
print('Created a preserved-candidate precision refinement; no checkpoint or evidence overwritten.')
