"""Train-only paired-goal capacity probe through the complete discrete rule path.

Cached features stop at grounding. All predictions use generated programs and
hard predicate tokens. This is not a closed-loop navigation success evaluation.
"""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--task-style',choices=['full','goal_only'],default='full')
    parser.add_argument('--pair-ranking-weight',type=float,default=0.)
    parser.add_argument('--task-token-encoder',action='store_true')
    parser.add_argument('--visual-source')
    parser.add_argument('--branch-tree',action='store_true')
    parser.add_argument('--route-temperature',type=float,default=1.)
    parser.add_argument('--spatial-encoding',action='store_true')
    parser.add_argument('--counterfactual-symbol-bc',action='store_true')
    parser.add_argument('--conjunctive-rules',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,args.root)
    from nesy_vla.runtime import read,setup,atomic,image_tensor,load_checkpoint
    from nesy_vla.model import NavigationModel
    from nesy_vla.symbols import tokenize
    from nesy_vla.logic import NoSupportedAction
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);data=Path(args.data)
    cfg=read(Path(args.root)/'configs/original.json');setup(cfg);torch.set_num_threads(2);torch.manual_seed(1011)
    if args.branch_tree:cfg['program']['action_semantics']='branch_tree'
    if args.conjunctive_rules:
        assert not args.branch_tree
        cfg['program']['action_semantics']='conjunctive_rules'
    cfg['program']['route_temperature']=args.route_temperature
    if args.spatial_encoding:cfg['spatial_encoding']=True
    model=NavigationModel(cfg).eval();source=load_checkpoint(args.checkpoint,model);source_update=source['update'];del source
    visual_transfer=None
    if args.visual_source:
        from nesy_vla.visual_transfer import initialize_visual_evidence
        visual_transfer=initialize_visual_evidence(args.visual_source,model)
    assert cfg['discrete_reasoning'] and cfg['generator_backend']=='symbolic_gru'
    # Freeze only the cached RGB encoder; query/truth, program synthesis and logic remain trainable.
    for module in [model.grounder.vision,model.grounder.projection]:module.requires_grad_(False)
    model.grounder.position.requires_grad_(False)
    records=read(data/'pairs.json');provenance=read(data/'source.json');allowed=set(provenance['train_layouts'])
    records=[r for r in records if r['both_successful'] and r['same_initial_rgb'] and r['first_actions_differ']]
    assert len(records)>=8,'Need at least eight confirmed contrasting pairs'
    assert all(r['scene'] in allowed for r in records)
    scenes=sorted({r['scene'] for r in records});heldout=set(scenes[::4])
    examples=[]
    for r in records:
        branches=r['branches'];images=[data/b['frames'][0]['image'] for b in branches]
        assert images[0].read_bytes()==images[1].read_bytes()
        for b,path in zip(branches,images):assert hashlib.sha256(path.read_bytes()).hexdigest()==b['frames'][0]['sha256']
        with torch.no_grad():tokens=model.grounder.encode(image_tensor(Image.open(images[0]).convert('RGB'),cfg['image_size'],'cpu')).detach()
        examples.append({'id':r['pair_id'],'scene':r['scene'],'tokens':tokens,
            'tasks':[b['episode']['category'] if args.task_style=='goal_only' else
                'Find the '+b['episode']['category']+' and stop close to it.' for b in branches],
            'labels':[b['frames'][0]['action'] for b in branches]})
    training=[r for r in examples if r['scene'] not in heldout];validation=[r for r in examples if r['scene'] in heldout]
    assert training and validation
    if args.task_token_encoder:
        from nesy_vla.task_token_generator import TaskTokenProgramGenerator
        vocabulary=sorted({task for row in training for task in row['tasks']})
        model.generator=TaskTokenProgramGenerator.initialize_from(model.generator,cfg['program'],vocabulary,cfg.get('generator_hidden',64))
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=.01);rng=random.Random(1011);started=time.time()
    def forward(rows,intervention=None):
        tasks=rows[0]['tasks'];programs=[model.generator(task,[],[]) for task in tasks];results=[];losses=[];ranking=[]
        for j,row in enumerate(rows):
            paired_probabilities=[]
            for goal,p in enumerate(programs):
                tokens=rows[(j+1)%len(rows)]['tokens'] if intervention=='image_swap' else row['tokens']
                if intervention=='patch_mirror':
                    from nesy_vla.perception import image_grid_encoding
                    position=model.grounder.position[None]
                    if cfg.get('spatial_encoding',False):
                        position=position+.5*image_grid_encoding(7,cfg['program']['d'],tokens.device)[None]
                    order=torch.arange(49,device=tokens.device).reshape(7,7).flip(1).flatten()
                    tokens=(tokens-position)[:,order]+position
                result=model.step(p,None,[],tokens=tokens)
                if intervention=='predicate_zero':
                    try:result=model.logic.infer(p,tokenize(torch.zeros(cfg['program']['k'])))
                    except NoSupportedAction:
                        # A counterfactual state may have no applicable rule.
                        # Record abstention; never invent or execute a fallback.
                        result={'probs':torch.zeros(cfg['program']['a']),'unsupported':True}
                loss=-result['probs'][row['labels'][goal]].clamp_min(1e-9).log()
                if args.counterfactual_symbol_bc and intervention is None:
                    from nesy_vla.counterfactual_learning import counterfactual_symbol_bc
                    from nesy_vla.symbols import DiscreteState
                    state=DiscreteState(result['symbol_codes'],result['symbol_values'])
                    loss=counterfactual_symbol_bc(model.logic,p,state,row['labels'][goal])
                losses.append(loss);results.append((result,row['labels'][goal]))
                paired_probabilities.append(result['probs'])
            if args.pair_ranking_weight:
                a,b=row['labels'];left,right=[p.clamp_min(1e-9).log() for p in paired_probabilities]
                # Counterfactual goals share the same RGB. No predicate identity
                # is prescribed; all four likelihoods come from generated rules.
                ranking.append(torch.nn.functional.softplus(2.-(left[a]-left[b]+right[b]-right[a])))
        loss=torch.stack(losses).mean()
        if ranking:loss=loss+args.pair_ranking_weight*torch.stack(ranking).mean()
        return loss,results
    @torch.no_grad()
    def measure(rows):
        def action(r):return -1 if r.get('unsupported',False) else int(r['probs'].argmax())
        loss,result=forward(rows);pred=[int(r['probs'].argmax()) for r,_ in result]
        labels=[a for _,a in result];pairs=len(rows)
        swapped=[pred[j^1] for j in range(len(pred))]
        codes=torch.stack([r['symbol_codes'] for r,_ in result]).reshape(pairs,2,-1)
        zero=forward(rows,'predicate_zero')[1];images=forward(rows,'image_swap')[1]
        mirror=forward(rows,'patch_mirror')[1]
        return {'pairs':pairs,'accuracy':sum(p==a for p,a in zip(pred,labels))/len(labels),
            'both_correct':sum(pred[2*i]==labels[2*i] and pred[2*i+1]==labels[2*i+1] for i in range(pairs)),
            'different_goal_changes_action':sum(pred[2*i]!=pred[2*i+1] for i in range(pairs)),
            'goal_swap_accuracy':sum(p==a for p,a in zip(swapped,labels))/len(labels),
            'image_swap_accuracy':sum(action(r)==a for r,a in images)/len(labels),
            'zero_predicate_accuracy':sum(action(r)==a for r,a in zero)/len(labels),
            'zero_predicate_abstentions':sum(r.get('unsupported',False) for r,_ in zero),
            'zero_predicate_changes_action':sum(action(r)!=p for (r,_),p in zip(zero,pred)),
            'mirror_patch_changes_action':sum(action(r)!=p for (r,_),p in zip(mirror,pred)),
            'mean_goal_symbol_bit_difference':float((codes[:,0]!=codes[:,1]).float().mean()),
            'nll':sum(float(-r['probs'][a].clamp_min(1e-9).log()) for r,a in result)/len(result),
            'objective':float(loss),'labels':labels,'predictions':pred}
    result={'source':args.checkpoint,'source_commit':read(Path(args.checkpoint)/'COMMITTED'),'source_update':source_update,
        'training_layouts':sorted({r['scene'] for r in training}),'probe_holdout_layouts':sorted(heldout),
        'fixed_validation_used':False,'navigation_success_test':False,'all_actions_via_discrete_rule_inference':True,
        'task_style':args.task_style,'pair_ranking_weight':args.pair_ranking_weight,
        'task_token_encoder':args.task_token_encoder,
        'visual_transfer':visual_transfer,
        'action_semantics':cfg['program'].get('action_semantics','horn'),
        'route_temperature':args.route_temperature,
        'spatial_encoding':bool(cfg.get('spatial_encoding',False)),
        'counterfactual_symbol_bc':args.counterfactual_symbol_bc,
        'dataset_provenance':provenance,'initial':{'train':measure(training),'heldout':measure(validation)},'learning':[]}
    atomic(out/'result.json',dict(result,state='running',pid=os.getpid(),time=time.time()))
    for step in range(1,args.steps+1):
        batch=rng.sample(training,min(4,len(training)));loss,_=forward(batch)
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,2.)
        if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        if step%20==0:
            row={'step':step,'seconds':time.time()-started,'unclipped_gradient_norm':float(norm),
                 'train':measure(training),'heldout':measure(validation)}
            result['learning'].append(row);atomic(out/'result.json',dict(result,state='running',pid=os.getpid(),time=time.time()))
            print(json.dumps({'step':step,'train_accuracy':row['train']['accuracy'],'heldout_accuracy':row['heldout']['accuracy']}),flush=True)
    atomic(out/'result.json',dict(result,state='complete',pid=os.getpid(),time=time.time(),seconds=time.time()-started))


if __name__=='__main__':main()
