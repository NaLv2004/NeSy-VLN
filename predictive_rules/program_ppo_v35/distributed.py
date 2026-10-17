"""Synchronous two-GPU PPO; globally weighted programs, not rank averages."""
import os
import torch
import torch.distributed as dist


def initialize():
    torch.cuda.set_device(0)  # Each subprocess sees exactly its assigned physical GPU.
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        dist.init_process_group('nccl')


def rank():return dist.get_rank() if dist.is_initialized() else 0
def world():return dist.get_world_size() if dist.is_initialized() else 1


def total(value):
    value=torch.as_tensor(value,device='cuda',dtype=torch.float64).clone()
    if dist.is_initialized():dist.all_reduce(value)
    return value


def maximum(value):
    value=torch.as_tensor(value,device='cuda',dtype=torch.float64).clone()
    if dist.is_initialized():dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return value


def synchronize_gradients(parameters):
    if dist.is_initialized():
        for parameter in parameters:
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)  # Losses already divided by global count.


def gather(value):
    if not dist.is_initialized():return [value]
    values=[None]*world();dist.all_gather_object(values,value);return values


def barrier():
    if dist.is_initialized():dist.barrier()


def aggregate(rows):
    result={};counts=['episodes','successes','executed_steps','rules_generated','planning_failures','false_stops','program_decisions']
    for key in counts:
        if all(key in row for row in rows):result[key]=sum(row[key] for row in rows)
    for key in set.intersection(*(set(r) for r in rows)):
        if key in result:continue
        if key=='by_group':
            groups={}
            for row in rows:
                for name,value in row[key].items():
                    item=groups.setdefault(name,dict(episodes=0,successes=0))
                    item['episodes']+=value['episodes'];item['successes']+=value['successes']
            for group in groups.values():group['success_rate']=group['successes']/group['episodes']
            result[key]=groups
        elif key in ['seconds','collection_seconds','optimization_seconds','elapsed_seconds','memory_peak_gib']:
            result[key]=max(r[key] for r in rows)
        elif isinstance(rows[0][key],bool):result[key]=all(r[key] for r in rows)
        elif isinstance(rows[0][key],(float,int)):result[key]=sum(r[key] for r in rows)/len(rows)
    result['success_rate']=result['successes']/max(1,result['episodes'])
    result['executed_steps_per_program']=result['executed_steps']/max(1,result['rules_generated'])
    result['five_step_average_requirement_met']=result['executed_steps']>=5*result['rules_generated']
    result['collision_action_rate']=sum(r['collision_action_rate']*r['executed_steps'] for r in rows)/max(1,result['executed_steps'])
    return result
