"""Training-only physical outcomes; probes cannot influence policy choices."""
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
                op=req['op']
                if op=='init':
                    env=NavigationEnvironment(req['experiment'],0,False);train=set(req['train_scenes'])
                    env.oracle_action=lambda *_:(_ for _ in ()).throw(RuntimeError('Oracle forbidden'))
                    result=True
                elif op=='reset':
                    e=req['episode'];assert e['scene'] in train and e['domain']=='normal' and e['distance_group'] in ('near','medium')
                    rgb=env.reset(e);result=dict(rgb=image(rgb),eligible=bool(env.private_metrics()['eligible']))
                elif op=='step':
                    root=env.state();frames=[];labels=[];first=[];second=[]
                    current=image(env.observe()['rgb'])
                    if req['probe']:
                        rng=np.random.default_rng(req['seed'])
                        try:
                            for a in range(17):
                                state,rgb,metrics=env.probe(a)
                                frames.append(image(rgb));labels.append(bool(metrics['eligible']));first.append(a);second.append(-1)
                                if a:
                                    env.restore(state);b=int(rng.integers(1,17))
                                    _,rgb,metrics=env.probe(b)
                                    frames.append(image(rgb));labels.append(bool(metrics['eligible']));first.append(a);second.append(b)
                                    env.restore(root)
                        finally:env.restore(root)
                        assert np.array_equal(current,image(env.observe()['rgb'])),'Training probes changed policy state'
                    a=req['action']
                    if a is None:actual=current;done=True;eligible=bool(env.private_metrics()['eligible']);success=False
                    else:
                        response=env.step(int(a));actual=image(response['rgb']);done=bool(response['done'])
                        eligible=bool(env.private_metrics()['eligible']);success=bool(response.get('metrics',{}).get('success',False))
                    result=dict(actual=actual,done=done,eligible=eligible,success=success,
                        future_images=np.stack(frames) if frames else np.empty((0,224,224,3),np.uint8),
                        future_labels=np.asarray(labels,np.bool_),first_action=np.asarray(first,np.int16),second_action=np.asarray(second,np.int16))
                elif op=='close':send(out,dict(ok=True,result=None));break
                else:raise ValueError(op)
                send(out,dict(ok=True,result=result))
            except Exception:send(out,dict(ok=False,error=traceback.format_exc()))
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
