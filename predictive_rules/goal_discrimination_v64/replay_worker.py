"""Replay TRAIN trajectories and return RGB plus the existing STOP outcome.

No geometry or semantic state assignment leaves this private simulator process.
The feedback is a training label only; no live policy receives it.
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
                    env.oracle_action=lambda *_:(_ for _ in ()).throw(RuntimeError('Oracle policy forbidden'))
                    checked=False;result=True
                elif op=='replay':
                    assert req['episode']['scene'] in train
                    env.reset(req['episode'])
                    if not checked:
                        expected=env.private_metrics()['eligible'];actual=env.step(0)
                        assert expected==bool(actual['metrics']['success'])
                        checked=True;env.reset(req['episode'])
                    frames=[image(env.observe()['rgb'])];labels=[env.private_metrics()['eligible']]
                    for action in req['actions']:
                        assert 0<=int(action)<17
                        response=env.step(int(action))
                        if int(action)==0:assert bool(response['metrics']['success'])==labels[-1]
                        frames.append(image(response['rgb']));labels.append(env.private_metrics()['eligible'])
                    result=dict(images=np.stack(frames),stop_success=np.asarray(labels,dtype=np.bool_),
                        steps=len(req['actions']),stop_label_matches_environment=True)
                elif op=='close':
                    send(out,dict(ok=True,result=None));break
                else:raise ValueError(op)
                send(out,dict(ok=True,result=result))
            except Exception:send(out,dict(ok=False,error=traceback.format_exc()))
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
