"""Offline expressivity diagnostic; free symbol codes are never a deployed policy."""
import os,sys,json,argparse,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.program import Program
from nesy_vla.neural_logic import NeuralSymbolicLogic
from nesy_vla.symbols import tokenize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    parser.add_argument('--steps',type=int,default=160);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.folder);torch.set_num_threads(2)
    trace=json.loads((folder/'trajectory.json').read_text());rows=trace['steps']
    assert rows and all(r['expert'] for r in rows) and trace['episode_id'].startswith('P25_train_')
    programs={}
    for file in folder.glob('program_*.json'):
        raw=json.loads(file.read_text())['program']
        p=Program.from_record(raw);programs[p.version]=p
    p=next(iter(programs.values()));logic=NeuralSymbolicLogic(len(p.definitions),p.action_heads.shape[-1],len(p.weights))
    logic.load_state_dict(torch.load(folder/'logic_snapshot.pt',map_location='cpu',weights_only=True));logic.requires_grad_(False)
    labels=torch.tensor([r['action'] for r in rows]);n=len(rows);best=[];started=time.time()
    for seed in range(3):
        torch.manual_seed(seed)
        logits=torch.nn.Parameter(torch.randn(n,len(p.definitions))*.2)
        optimizer=torch.optim.Adam([logits],lr=.04);peak=0;last=None
        for step in range(args.steps):
            probs=torch.stack([logic.infer(programs[r['version']],tokenize(z.sigmoid()))['probs'] for r,z in zip(rows,logits)])
            accuracy=float((probs.argmax(-1)==labels).float().mean());peak=max(peak,accuracy)
            loss=-probs[torch.arange(n),labels].clamp_min(1e-8).log().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            last={'accuracy':accuracy,'nll':float(loss.detach()),'predictions':probs.argmax(-1).tolist()}
        best.append({'seed':seed,'best_accuracy_observed':peak,'final':last})
    result={'episode_id':trace['episode_id'],'expert_labels':labels.tolist(),'steps':n,
        'meaning':'fixed automatic rules with separately optimized Boolean states; not navigation success',
        'trials':best,'seconds':time.time()-started}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
