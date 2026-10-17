"""Distributed optimization of a single symbolic actor; no extra action pathway."""
import torch
import torch.distributed as dist


def global_moments(values):
    x=values.detach().double()
    stats=torch.stack([x.sum(),x.square().sum(),x.new_tensor(x.numel())])
    dist.all_reduce(stats)
    mean=stats[0]/stats[2].clamp_min(1)
    variance=(stats[1]/stats[2].clamp_min(1)-mean.square()).clamp_min(0)
    return mean.to(values),variance.sqrt().clamp_min(1e-6).to(values)


def average_gradients(parameters):
    """Identical ordered collectives even when an auxiliary parameter is unused."""
    parameters=[p for p in parameters if p.requires_grad]
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    dist.all_reduce(flat);flat.div_(dist.get_world_size())
    offset=0
    for parameter in parameters:
        count=parameter.numel();grad=flat[offset:offset+count].reshape_as(parameter)
        if parameter.grad is None:parameter.grad=grad.clone()
        else:parameter.grad.copy_(grad)
        offset+=count


def gather_objects(value):
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value);return values


@torch.no_grad()
def assert_identical_parameters(module):
    """Bitwise equality to rank zero, not a loose aggregate checksum."""
    flat=torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    reference=flat.clone();dist.broadcast(reference,src=0)
    error=(flat-reference).abs().max();dist.all_reduce(error,op=dist.ReduceOp.MAX)
    if float(error)!=0:raise RuntimeError(f'Distributed parameter divergence: {float(error)}')
    return float(error)
