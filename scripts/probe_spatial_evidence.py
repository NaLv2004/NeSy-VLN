"""Measure whether the predicate reader uses patch location as well as content."""
import argparse,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from nesy_vla.program import Program
from nesy_vla.runtime import read,atomic,image_tensor,sha
from probe_loss_gradients import FixedProgramModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--trace',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();torch.set_num_threads(2)
    folder=Path(a.checkpoint);marker=read(folder/'COMMITTED');path=folder/marker['checkpoint_file']
    assert sha(path)==marker['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False);cfg=state['config']
    model=FixedProgramModel(cfg,state);del state
    trace_folder=Path(a.trace);trace=read(trace_folder/'trajectory.json');rows=trace['steps']
    assert trace['episode_id'].startswith('P25_train_') and all(r['expert'] for r in rows)
    images=[Image.open(trace_folder/r['image']).convert('RGB') for r in rows]
    programs={}
    for f in trace_folder.glob('program_*.json'):
        data=read(f)['program'];programs[data['version']]=Program.from_record(data)
    tensor=torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images])
    with torch.no_grad():
        tokens=model.grounder.encode(tensor);position=model.grounder.position[None]
        content=tokens-position;mirror=torch.arange(49).reshape(7,7).flip(1).flatten()
        variants={'original':tokens,'no_position':content,'mirror_content_keep_positions':content[:,mirror]+position}
        predictions={};beliefs={}
        for name,features in variants.items():
            bs=[];qs=[]
            for index,row in enumerate(rows):
                p=programs[row['version']];b,c=model.grounder.observe(features[index:index+1],p.definitions,[])
                bs.append(b);qs.append(model.logic.infer(p,b,c)['probs'])
            beliefs[name]=torch.stack(bs);predictions[name]=torch.stack(qs)
        original=predictions['original'];result={'episode':trace['episode_id'],'steps':len(rows),
            'checkpoint':str(path),'scope':'saved training programs; visual predicate reader with no history/prior',
            'content_rms':float(content.square().mean().sqrt()),'position_rms':float(position.square().mean().sqrt()),
            'position_content_rms_ratio':float(position.square().mean().sqrt()/content.square().mean().sqrt()),
            'original_actions':original.argmax(-1).tolist(),'expert_actions':[r['action'] for r in rows],
            'ablations':{name:{'probability_tv':float((values-original).abs().sum(-1).mean()/2),
                'changed_actions':int((values.argmax(-1)!=original.argmax(-1)).sum()),
                'truth_mean_absolute_change':float((beliefs[name]-beliefs['original']).abs().mean())}
                for name,values in predictions.items() if name!='original'}}
    atomic(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
