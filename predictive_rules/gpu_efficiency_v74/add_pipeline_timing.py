"""Speed independent simulator startup and expose end-to-end timing."""
from pathlib import Path
import ast
p=Path(__file__).parent.parent/'expanded_dynamics_v71/evaluate.py';s=p.read_text()
old="        sims=[bank.simulator(out/f'sim_rank{rank}_{i}.log',int(os.environ.get('LOCAL_RANK','0')),evaluation=True) for i in range(workers)]"
assert old in s
s=s.replace(old,"""        startup=time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(8,workers)) as startup_pool:
            sims=list(startup_pool.map(lambda i:bank.simulator(out/f'sim_rank{rank}_{i}.log',int(os.environ.get('LOCAL_RANK','0')),evaluation=True),range(workers)))
        atomic(out/f'rank{rank}.startup.json',dict(workers=workers,parallel_startup=8,seconds=time.perf_counter()-startup,time=time.time()))""")
s=s.replace('requests=[];prepared=[]','tick=time.perf_counter();requests=[];prepared=[]')
s=s.replace('                    extend_batch([programs[i] for i in active],32-step,device)',
'''                    neural_seconds=time.perf_counter()-tick;planner_tick=time.perf_counter()
                    extend_batch([programs[i] for i in active],32-step,device)
                    planner_seconds=time.perf_counter()-planner_tick;proof_tick=time.perf_counter()''')
s=s.replace('                    if not requests:break','''                    proof_seconds=time.perf_counter()-proof_tick
                    if not requests:break
                    environment_tick=time.perf_counter()''')
s=s.replace('                    active=[]\n                    for (i,action),response', '''                    environment_seconds=time.perf_counter()-environment_tick
                    with (out/f'rank{rank}.pipeline.jsonl').open('a') as log:
                        log.write(json.dumps(dict(time=time.time(),offset=offset,step=step,active=len(requests),
                            neural_seconds=neural_seconds,gpu_symbolic_seconds=planner_seconds,proof_seconds=proof_seconds,
                            environment_seconds=environment_seconds,total_seconds=time.perf_counter()-tick,
                            allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30))+'\\n')
                    active=[]
                    for (i,action),response''')
ast.parse(s);p.write_text(s)
print('Pending evaluation now starts simulators eight-at-a-time and records pipeline component times.')
