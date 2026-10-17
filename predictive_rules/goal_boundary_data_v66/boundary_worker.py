"""Measured one/two-action image outcomes; no state semantics exported."""
import os,sys,traceback
import numpy as np
from PIL import Image
sys.path.insert(0,'/home/user/xjy/strips_wm_navigation_20261009')
sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')

def main():
    from objectnav_rl.ipc import send,receive
    from stripsnav.sim_adapter import NavigationEnvironment
    out=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0);os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    env=None
    def image(rgb):return np.asarray(Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8)).resize((224,224),Image.Resampling.BILINEAR))
    try:
        while True:
            try:req=receive(sys.stdin.buffer)
            except EOFError:break
            try:
                if req['op']=='init':
                    env=NavigationEnvironment(req['experiment'],0,False);train=set(req['train_scenes'])
                    env.oracle_action=lambda *_:(_ for _ in ()).throw(RuntimeError('Oracle forbidden'))
                    result=True
                elif req['op']=='replay':
                    assert req['episode']['scene'] in train
                    env.reset(req['episode']);anchors=set(req['anchors']);frames=[];labels=[];steps=[];first=[];second=[]
                    rng=np.random.default_rng(req['seed']);one=two=0
                    def append(rgb,eligible,t,a,b):
                        frames.append(image(rgb));labels.append(bool(eligible));steps.append(t);first.append(a);second.append(b)
                    for t in range(len(req['actions'])+1):
                        if t in anchors:
                            root=env.state()
                            for a in range(17):
                                state,rgb,metrics=env.probe(a)
                                append(rgb,metrics['eligible'],t,a,-1);one+=1
                                # STOP terminates a rollout; only continuing
                                # operators are used as a two-step prefix.
                                if a:
                                    env.restore(state);b=int(rng.integers(1,17))
                                    try:
                                        _,rgb,metrics=env.probe(b)
                                        append(rgb,metrics['eligible'],t,a,b);two+=1
                                    finally:env.restore(root)
                            assert np.allclose(env.state().position,root.position)
                        if t==len(req['actions']):break
                        response=env.step(int(req['actions'][t]))
                        if int(req['actions'][t])==0:break
                    assert len(frames)==len(anchors)*33
                    result=dict(images=np.stack(frames),stop_success=np.asarray(labels,np.bool_),root_step=np.asarray(steps,np.int16),
                        first_action=np.asarray(first,np.int16),second_action=np.asarray(second,np.int16),
                        roots=sorted(anchors),one_step_probes=one,two_step_probes=two)
                elif req['op']=='close':send(out,dict(ok=True,result=None));break
                else:raise ValueError(req['op'])
                send(out,dict(ok=True,result=result))
            except Exception:send(out,dict(ok=False,error=traceback.format_exc()))
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
