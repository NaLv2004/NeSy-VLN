import unittest
from nesy_vla.batching import next_batch_size,pack_chunks


class Capacity(unittest.TestCase):
    def test_grow_only_with_useful_work_and_headroom(self):
        self.assertEqual(next_batch_size(14,14,1,24,30),16)
        self.assertEqual(next_batch_size(14,14,1,24,8),14)
        self.assertEqual(next_batch_size(14,22,1,24,30),14)

    def test_physical_rendering_memory_reduces_budget(self):
        self.assertLess(next_batch_size(14,21,3,24,30),14)
        self.assertEqual(next_batch_size(12,10,1,24,30,backed_off=True),12)

    def test_maximum(self):
        self.assertEqual(next_batch_size(32,10,1,24,40),32)

    def test_long_revisions_do_not_inherit_short_prompt_capacity(self):
        initial=[({'previous':None,'rows':[0]*2},False) for _ in range(32)]
        revisions=[({'previous':object(),'rows':[0]*10},False) for _ in range(27)]
        batches=pack_chunks(initial+revisions,32)
        self.assertEqual([len(b) for b in batches],[32,13,13,1])
        self.assertTrue(all(sum(len(c['rows']) for c,_ in b)<=130 for b in batches))
        self.assertEqual([x for b in batches for x in b],initial+revisions)
