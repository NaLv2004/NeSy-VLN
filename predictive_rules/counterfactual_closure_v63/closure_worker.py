"""Private two-action training probes, restoring the physical scene each time.

No coordinates, goal measurements, collision labels or oracle trajectories are
returned. The policy has already selected its real action from a symbolic proof.
"""
import os,sys,traceback
import numpy as np
from PIL import Image
sys.path.insert(0,'/home/user/xjy/strips_wm_navigation_20261009')
sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')

def main():
    from objectnav_rl.ipc import send,receive
    from stripsnav.sim_adapter import NavigationEnvironment
    out=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno());env=None
    def image(rgb):
        return np.asarray(Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8)).resize((224,224),Image.Resampling.BILINEAR))
    try:
        while True:
            try:req=receive(sys.stdin.buffer)
            except EOFError:break
            try:
                op=req['op']
                if op=='init':
                    env=NavigationEnvironment(req['experiment'],0,False);train=set(req['train_scenes'])
                    # Counterfactual labels need only rendered observations.
                    env.private_metrics=lambda obs=None:{}
                    env.oracle_action=lambda *_:(_ for _ in ()).throw(RuntimeError('Oracle forbidden'))
                    result=True
                elif op=='reset':
                    assert req['episode']['scene'] in train
                    initial=image(env.reset(req['episode']));errors=[]
                    for a,b in req.get('verify_pairs',[]):
                        state,_,_=env.probe(int(a));env.restore(state)
                        _,expected,_=env.probe(int(b))
                        env.reset(req['episode']);env.step(int(a));actual=env.step(int(b))
                        error=float(np.abs(image(expected).astype(np.float32)-image(actual['rgb']).astype(np.float32)).mean())
                        assert error<.5,dict(two_step_probe_error=error,actions=[a,b])
                        errors.append(error);env.reset(req['episode'])
                    assert np.array_equal(initial,image(env.observe()['rgb']))
                    result=dict(rgb=initial,two_step_checks=errors)
                elif op in ('probe_and_execute','probe_only'):
                    chosen=int(req['action']) if op=='probe_and_execute' else None
                    if chosen is not None:assert 0<=chosen<17
                    second=np.asarray(req['second_actions'],dtype=np.int64)
                    assert second.shape==(17,2) and second.min()>=0 and second.max()<17
                    original=env.state();current=image(env.observe()['rgb'])
                    first=[];twice=[]
                    try:
                        for a in range(17):
                            if a==0:state,rgb=original,current
                            else:
                                state,raw,_=env.probe(a);rgb=image(raw)
                            first.append(rgb)
                            env.restore(state)
                            for b in second[a]:
                                if a==0:
                                    # STOP is terminal: do not pretend a later
                                    # action was executable. Store a masked copy.
                                    twice.append(rgb)
                                elif b==0:twice.append(rgb)
                                else:
                                    _,raw,_=env.probe(int(b));twice.append(image(raw))
                            env.restore(original)
                    finally:env.restore(original)
                    assert np.array_equal(current,image(env.observe()['rgb'])),'Counterfactual probe changed actual state'
                    result=dict(current=current,successors=np.stack(first),second_successors=np.stack(twice).reshape(17,2,224,224,3),
                        second_valid=np.broadcast_to(np.arange(17)[:,None]!=0,(17,2)).copy(),executed=False)
                    if chosen is not None:
                        actual=env.step(chosen);rgb=image(actual['rgb'])
                        error=float(np.abs(rgb.astype(np.float32)-first[chosen].astype(np.float32)).mean())
                        result.update(actual=rgb,done=bool(actual['done']),executed=True,
                            action=chosen,probe_rgb_error=error,probe_matches_execution=error<.5)
                elif op=='close':
                    send(out,dict(ok=True,result=None));break
                else:raise ValueError(op)
                send(out,dict(ok=True,result=result))
            except Exception:send(out,dict(ok=False,error=traceback.format_exc()))
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
