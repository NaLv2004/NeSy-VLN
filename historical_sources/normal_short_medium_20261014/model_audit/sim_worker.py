"""IPC boundary prevents neural policies from receiving private geometry."""
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1")
from objectnav_rl.ipc import receive, send
from stripsnav.sim_adapter import NavigationEnvironment

output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
env = None
try:
    while True:
        try:
            request = receive(sys.stdin.buffer)
        except EOFError:
            break
        try:
            op = request["op"]
            if op == "init":
                env = NavigationEnvironment(request["experiment"], 0, request.get("evaluation", True))
                result = True
            elif op == "reset":
                result = {"rgb": env.reset(request["episode"]), "private_metrics": env.private_metrics()}
            elif op == "step":
                result = env.step(request["action"])
                # Depth remains private to the simulator. These RGB-only policies
                # do not need to transport a large unused depth array through IPC.
                result.pop('depth_gt', None)
            elif op == "close":
                send(output, {"ok": True, "result": None})
                break
            else:
                raise ValueError(op)
            send(output, {"ok": True, "result": result})
        except Exception:
            send(output, {"ok": False, "error": traceback.format_exc()})
finally:
    if env:
        env.close()
