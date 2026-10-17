"""Replay immutable training actions; expose only their current RGB frames."""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/user/xjy/strips_wm_navigation_20261009')
sys.path.insert(0, '/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')

def main():
    from objectnav_rl.ipc import receive, send
    from stripsnav.sim_adapter import NavigationEnvironment
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    env = None
    try:
        while True:
            try: request = receive(sys.stdin.buffer)
            except EOFError: break
            try:
                if request['op'] == 'init':
                    env = NavigationEnvironment(request['experiment'], 0, False)
                    allowed = set(request['train_scenes']); result = True
                elif request['op'] == 'replay':
                    ep = request['episode']
                    if ep['scene'] not in allowed: raise ValueError('Training-only replay')
                    actions = request['actions']
                    if not 1 <= len(actions) <= 32: raise ValueError('Invalid recorded length')
                    env.reset(ep); frames = []
                    for index, action in enumerate(actions):
                        rgb = np.asarray(env.observe()['rgb'])[..., :3].astype(np.uint8)
                        frames.append(np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR)))
                        # No policy, probes, rewards, state labels or action replacement.
                        response = env.step(int(action))
                        if response['done'] and index != len(actions) - 1:
                            raise ValueError('Recorded actions outlive the replay episode')
                    result = dict(rgb=np.stack(frames))
                elif request['op'] == 'close':
                    send(output, {'ok': True, 'result': None}); break
                else: raise ValueError(request['op'])
                send(output, {'ok': True, 'result': result})
            except Exception:
                import traceback
                send(output, {'ok': False, 'error': traceback.format_exc()})
    finally:
        if env is not None: env.close()

if __name__ == '__main__': main()
