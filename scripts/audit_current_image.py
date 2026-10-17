"""Replay frozen development traces and ablate only the latest visual evidence.

All action probabilities still come from the generated rules and discrete
symbols. No simulator calls, training, geometric labels or map features occur.
"""
import argparse,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from nesy_vla.runtime import setup,load_checkpoint,image_tensor
from nesy_vla.model import NavigationModel
from nesy_vla.program import Program


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--step',type=int,default=10);parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--cpu-bf16',action='store_true');parser.add_argument('--first-only',action='store_true')
    parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(2)
    run=Path(args.run);cfg=json.loads((run/'config.json').read_text());setup(cfg)
    assert cfg['generator_backend']=='symbolic_gru' and cfg['discrete_reasoning']
    model=NavigationModel(cfg).eval()
    folder=run/f'evaluations/step_{args.step:06d}'
    load_checkpoint(folder/'checkpoint',model)
    paths=sorted((folder/'development').glob('*/trajectory.json'))
    # Evenly spaced IDs across the completed development suite, not selected by success.
    paths=[paths[i*len(paths)//min(args.episodes,len(paths))] for i in range(min(args.episodes,len(paths)))]
    results=[]
    for path in paths:
        raw=json.loads(path.read_text());programs={}
        for p in path.parent.glob('program_*.json'):
            record=json.loads(p.read_text())['program']
            program=Program.from_record(record)
            programs[program.version]=program
        rows=raw['steps'][:1] if args.first_only else raw['steps']
        images=[Image.open(path.parent/r['image']).convert('RGB') for r in rows]
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=args.cpu_bf16):
            tokens=model.grounder.encode(torch.cat([image_tensor(image,cfg['image_size'],'cpu') for image in images]))
        memory=[];prior=None;version=None;counts={'steps':len(rows),'fixed_latest_image_action_changes':0,
            'no_perceptual_history_action_changes':0,'cpu_saved_action_mismatch':0,'cpu_saved_bit_mismatches':0,
            'fixed_latest_image_bit_changes':0,'total_bits':0,'fixed_image_probability_tv_sum':0.,'saved_observation_absolute_error_sum':0.}
        for index,(row,image) in enumerate(zip(rows,images)):
            if version!=row['version']:prior=None;version=row['version']
            program=programs[version]
            actual=model.step(program,image,memory,prior,tokens=tokens[index:index+1])
            fixed=model.step(program,images[0],memory,prior,tokens=tokens[:1])
            no_history=model.step(program,image,[],prior,tokens=tokens[index:index+1])
            choice=actual['probs'].argmax()
            counts['fixed_latest_image_action_changes']+=int(choice!=fixed['probs'].argmax())
            counts['no_perceptual_history_action_changes']+=int(choice!=no_history['probs'].argmax())
            counts['cpu_saved_action_mismatch']+=int(choice!=torch.tensor(row['support']).argmax())
            counts['cpu_saved_bit_mismatches']+=int((actual['symbol_codes']!=torch.tensor(row['symbol_codes'])).sum())
            counts['fixed_latest_image_bit_changes']+=int((actual['symbol_codes']!=fixed['symbol_codes']).sum())
            counts['total_bits']+=actual['symbol_codes'].numel()
            counts['fixed_image_probability_tv_sum']+=float((actual['probs']-fixed['probs']).abs().sum()/2)
            counts['saved_observation_absolute_error_sum']+=float((actual['observation']-torch.tensor(row['observed_truth'])).abs().sum())
            prior,memory,_=model.advance(program,actual,row['action'],row['feedback'],memory)
        results.append(dict(counts,episode_id=raw['episode_id']))
    totals={key:sum(row[key] for row in results) for key in counts}
    result={'run':str(run),'step':args.step,'split':'development','episodes':results,'totals':totals,
        'qualification':'CPU replay versus original GPU replay can differ near Boolean thresholds; mismatch counts are reported. These ablations are not navigation success rates.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
