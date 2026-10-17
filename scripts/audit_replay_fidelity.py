"""Compare recorded JPEGs against fresh lossless reset observations, no policy rollout."""
import json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='6'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from nesy_vla.runtime import setup,load_checkpoint,image_tensor,DataBank,pil
from nesy_vla.model import NavigationModel
from nesy_vla.program import Program
torch.set_num_threads(2)
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_sft_v5');run=root/'runs/sft_plain'
cfg=json.loads((run/'config.json').read_text());setup(cfg);model=NavigationModel(cfg).eval()
folder=run/'evaluations/step_000010';load_checkpoint(folder/'checkpoint',model)
bank=DataBank(cfg);episodes={p['episode_id']:p for p in bank.dev}
out=root/'diagnostics/replay_fidelity10';out.mkdir(parents=True,exist_ok=True)
sim=bank.simulator(out/'render.log',0);records=[]
try:
    for rank in range(3):
        episode_id=json.loads((folder/'development'/f'rank{rank}.json').read_text())[0]['episode_id']
        trace_folder=folder/'development'/episode_id;trace=json.loads((trace_folder/'trajectory.json').read_text());row=trace['steps'][0]
        old=Image.open(trace_folder/row['image']).convert('RGB')
        actual=pil(sim.call('reset',episode=episodes[episode_id])['rgb']);actual.save(out/f'rank{rank}_lossless.png')
        record=json.loads((trace_folder/'program_000.json').read_text())['program']
        program=Program.from_record(record)
        snapshot=torch.load(trace_folder/'logic_snapshot.pt',weights_only=True,map_location='cpu')
        differences={k:float((v-model.logic.state_dict()[k]).abs().max()) for k,v in snapshot.items()}
        cases={}
        with torch.no_grad():
            for name,image in [('recorded_jpeg',old),('lossless_reset',actual)]:
                result=model.step(program,image,[],None)
                cases[name]={'truth_mae':float((result['observation']-torch.tensor(row['observed_truth'])).abs().mean()),
                    'truth':result['observation'].tolist(),'codes':result['symbol_codes'].tolist(),
                    'action':int(result['probs'].argmax()),'saved_action':row['action']}
        records.append({'episode_id':episode_id,'rank':rank,'logic_snapshot_max_error':max(differences.values()),
            'image_pixel_mae':float(np.abs(np.asarray(actual).astype(float)-np.asarray(old).astype(float)).mean()),
            'cases':cases,'saved_truth':row['observed_truth']})
finally:sim.close()
(out/'result.json').write_text(json.dumps(records,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ['saved_truth','cases']}|
    {'cases':{k:{q:v for q,v in c.items() if q not in ['truth','codes']} for k,c in r['cases'].items()}} for r in records],indent=2))
