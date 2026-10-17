from nesy_vla.resources import physical_gpu


def test_single_gpu_renderer_uses_the_assigned_physical_card():
    assert physical_gpu(0,'6')=='6'
    assert physical_gpu(0,'7')=='7'


def test_noncontiguous_worker_mapping_and_uuid_are_preserved():
    assert physical_gpu(1,'2,5,7')=='5'
    assert physical_gpu(0,'GPU-example-uuid')=='GPU-example-uuid'


def test_complete_demonstration_epochs_do_not_duplicate_or_drop_samples():
    from nesy_vla.resources import demonstration_schedule
    ids=list(range(23));all_ids=[]
    for step in range(1,5):
        slices=[demonstration_schedule(ids,step,rank,3,4,8) for rank in range(3)]
        all_ids.extend(x for rows in slices for x in rows)
    assert sorted(all_ids[:23])==ids
    assert sorted(all_ids[23:46])==ids
    assert demonstration_schedule(ids,2,1,3,4,8)==demonstration_schedule(ids,2,1,3,4,8)
