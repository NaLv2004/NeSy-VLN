"""Private train-only physical experiments; no policy, map, or state labels."""
import os
import sys
import traceback
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

    def image(rgb):
        return np.asarray(Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).resize(
            (224, 224), Image.Resampling.BILINEAR))

    try:
        while True:
            try:
                request = receive(sys.stdin.buffer)
            except EOFError:
                break
            try:
                op = request['op']
                if op == 'init':
                    env = NavigationEnvironment(request['experiment'], 0, False)
                    train = set(request['train_scenes'])
                    result = True
                elif op == 'reset':
                    if request['episode']['scene'] not in train:
                        raise ValueError('Only fixed training layouts may be probed')
                    result = dict(rgb=image(env.reset(request['episode'])))
                elif op in ('probe_and_execute', 'probe_only'):
                    # A proof-derived choice has already been committed by the
                    # policy before this request. No probe can affect it.
                    chosen = int(request['action']) if op == 'probe_and_execute' else None
                    if chosen is not None:
                        assert 0 <= chosen < 17
                    current = image(env.observe()['rgb'])
                    outcomes = [current]
                    for action in range(1, 17):
                        _, rgb, _ = env.probe(action)
                        outcomes.append(image(rgb))
                    assert np.array_equal(current, image(env.observe()['rgb'])), 'Probe changed current observation'
                    result = dict(current=current, successors=np.stack(outcomes), executed=False)
                    if chosen is not None:
                        response = env.step(chosen)
                        actual = image(response['rgb'])
                        error = float(np.abs(actual.astype(np.float32) - outcomes[chosen].astype(np.float32)).mean())
                        result.update(actual=actual, done=bool(response['done']), executed=True,
                                      action=chosen, probe_rgb_error=error,
                                      probe_matches_execution=bool(error < .5))
                    # No distance, reward, geometry, collision or target mask
                    # leaves this worker. Observations are the only labels.
                elif op == 'close':
                    send(output, dict(ok=True, result=None))
                    break
                else:
                    raise ValueError(op)
                send(output, dict(ok=True, result=result))
            except Exception:
                send(output, dict(ok=False, error=traceback.format_exc()))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
