"""Resolve a worker-local device slot without losing its physical GPU mapping."""
import os
import random


def physical_gpu(local_index,visible=None):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    if visible is None:return str(local_index)
    devices=[s.strip() for s in visible.split(',') if s.strip()]
    if not 0<=local_index<len(devices):raise ValueError('Local GPU index is outside the visible device list')
    return devices[local_index]


def demonstration_schedule(ids,update,rank,world,batch,seed):
    """Disjoint rank slices of reproducibly shuffled complete demonstration epochs."""
    start=(update-1)*world*batch+rank*batch;result=[];permutations={}
    for index in range(start,start+batch):
        epoch,offset=divmod(index,len(ids))
        if epoch not in permutations:
            order=list(ids);random.Random(seed+1009*epoch).shuffle(order);permutations[epoch]=order
        result.append(permutations[epoch][offset])
    return result


def paired_demonstration_schedule(base_ids,pairs,update,rank,world,batch,seed,fraction):
    """Keep both successful branches of each RGB-matched task in a training update."""
    if not 0<fraction<=1:raise ValueError('Paired fraction must be in (0,1]')
    count=int(batch*fraction)//2
    if count<1 or not pairs:raise ValueError('Paired supervision requires at least one whole pair per rank')
    if any(len(p)!=2 or p[0]==p[1] for p in pairs):raise ValueError('Expected distinct episode pairs')
    paired_ids={x for pair in pairs for x in pair}
    if paired_ids.intersection(base_ids):raise ValueError('Base and paired demonstration IDs overlap')
    selected=demonstration_schedule(pairs,update,rank,world,count,seed+7001)
    remaining=batch-2*count
    if remaining and not base_ids:raise ValueError('Missing base demonstrations')
    base=demonstration_schedule(base_ids,update,rank,world,remaining,seed) if remaining else []
    return base+[episode for pair in selected for episode in pair]


def recovery_demonstration_schedule(base_ids,pairs,recovery_ids,update,rank,world,batch,seed,pair_fraction,recovery_fraction):
    if not 0<recovery_fraction<1 or not 0<=pair_fraction<1 or pair_fraction+recovery_fraction>1:
        raise ValueError('Invalid demonstration mixture fractions')
    paired_ids={eid for pair in pairs for eid in pair}
    if set(recovery_ids)&(set(base_ids)|paired_ids):raise ValueError('Demonstration sources overlap')
    count=int(batch*recovery_fraction)
    if count<1 or not recovery_ids:raise ValueError('Recovery supervision needs a nonempty slice')
    remaining=batch-count
    normal=(paired_demonstration_schedule(base_ids,pairs,update,rank,world,remaining,seed,
        batch*pair_fraction/remaining) if pair_fraction else
        demonstration_schedule(base_ids,update,rank,world,remaining,seed))
    return normal+demonstration_schedule(recovery_ids,update,rank,world,count,seed+13001)
