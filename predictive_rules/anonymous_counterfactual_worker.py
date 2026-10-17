"""Private training-only simulator probes. No neural model is imported here."""
import os,sys,json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,"/home/user/xjy/strips_wm_navigation_20261009")
sys.path.insert(0,"/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1")
def worker():
    from objectnav_rl.ipc import receive,send
    from stripsnav.sim_adapter import NavigationEnvironment
    output=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0);os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    env=None
    def image(rgb):
        return np.asarray(Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8)).resize((224,224),Image.Resampling.BILINEAR))
    try:
        while True:
            try:request=receive(sys.stdin.buffer)
            except EOFError:break
            try:
                if request['op']=='init':
                    env=NavigationEnvironment(request['experiment'],0,False)
                    train=set(request['train_scenes']);result=True
                elif request['op']=='reset':
                    if request['episode']['scene'] not in train:raise ValueError('Counterfactual collection is training-only')
                    result=dict(rgb=image(env.reset(request['episode'])))
                elif request['op']=='probe_and_execute':
                    # The actual demonstration action is fixed before probing.
                    chosen=int(request['action']);current=image(env.observe()['rgb'])
                    scores=np.full(17,-.02,np.float32)
                    scores[0]=1. if env.private_metrics()['eligible'] else -1.
                    images=[current]
                    for action in range(1,17):
                        _,rgb,metrics=env.probe(action)
                        images.append(image(rgb))
                    response=env.step(chosen);actual=image(response['rgb'])
                    error=float(np.abs(actual.astype(np.float32)-images[chosen].astype(np.float32)).mean())
                    result=dict(current=current,successors=np.stack(images),scalar_rewards=scores,
                        actual=actual,done=response['done'],probe_rgb_error=error,
                        probe_matches_execution=bool(error<.5),action=chosen)
                elif request['op']=='close':
                    send(output,{'ok':True,'result':None});break
                else:raise ValueError(request['op'])
                send(output,{'ok':True,'result':result})
            except Exception:
                import traceback
                send(output,{'ok':False,'error':traceback.format_exc()})
    finally:
        if env is not None:env.close()


if __name__=="__main__":worker()
