"""Adjust useful batch work against observed physical GPU capacity, never reserve dummy tensors."""
import math


def pack_chunks(chunks,limit,frame_budget=130,revision_limit=13):
    batches=[];batch=[];frames=0;kind=None
    for item in chunks:
        chunk,_=item;revision=chunk['previous'] is not None;n=len(chunk['rows'])
        capacity=min(limit,revision_limit) if revision else limit
        if batch and (kind!=revision or len(batch)>=capacity or frames+n>frame_budget):
            batches.append(batch);batch=[];frames=0
        batch.append(item);frames+=n;kind=revision
    if batch:batches.append(batch)
    return batches


def next_batch_size(current,peak_allocated,non_torch,total,chunk_count,maximum=32,backed_off=False):
    if backed_off:return current
    budget=max(0.,.97*total-non_torch)
    if peak_allocated>budget and current>1:
        return max(1,min(current-1,math.floor(current*budget/max(1.,peak_allocated))))
    if peak_allocated<.90*budget and chunk_count>current:
        return min(maximum,chunk_count,current+2)
    return current
