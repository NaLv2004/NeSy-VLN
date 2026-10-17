import unittest
from nesy_vla.resources import paired_demonstration_schedule


class PairedScheduleTests(unittest.TestCase):
    def test_keeps_both_branches_without_cross_rank_overlap(self):
        pairs=[(f'a{i}',f'b{i}') for i in range(50)];base=[f'base{i}' for i in range(200)]
        outputs=[paired_demonstration_schedule(base,pairs,1,rank,4,12,11,1/3) for rank in range(4)]
        self.assertEqual(len(set(x for rows in outputs for x in rows)),48)
        for rows in outputs:
            self.assertEqual(len(rows),12)
            selected=set(rows[8:]);self.assertEqual(sum(set(p).issubset(selected) for p in pairs),2)
    def test_reproducible_epochs_cover_all_pairs(self):
        pairs=[(f'a{i}',f'b{i}') for i in range(10)];base=['base']
        first=[paired_demonstration_schedule(base,pairs,u,0,1,4,21,1.) for u in range(1,6)]
        second=[paired_demonstration_schedule(base,pairs,u,0,1,4,21,1.) for u in range(1,6)]
        self.assertEqual(first,second);self.assertEqual(len(set(x for row in first for x in row)),20)
    def test_reject_overlap(self):
        with self.assertRaises(ValueError):paired_demonstration_schedule(['a'],[('a','b')],1,0,1,4,1,.5)
    def test_reject_incomplete_pair_capacity(self):
        with self.assertRaises(ValueError):paired_demonstration_schedule(['base'],[('a','b')],1,0,1,1,1,1.)


if __name__=='__main__':unittest.main()
