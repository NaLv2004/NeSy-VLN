"""Verify the new schema keeps the existing isolated examples exactly."""
import unittest
import numpy as np
from pathlib import Path
from predictive_nesy.data import load_training_windows,read,BASE
from predictive_nesy.anonymous_data import load_windows,canonical_provenance,batch,OutcomeBank
from predictive_nesy.anonymous_training import require_anonymous


class CorpusTest(unittest.TestCase):
    def test_same_windows_without_loading_semantic_supervision(self):
        manifest=read(BASE/'predictive_rules_20261015/run_anonymous_v5/manifest.json')
        directory=manifest['data']['source'];layouts=manifest['data']['train_layouts']
        old,old_provenance=load_training_windows(directory,layouts)
        new,provenance=load_windows(directory,layouts)
        self.assertEqual(provenance,canonical_provenance(old_provenance));self.assertEqual(len(old),len(new))
        for before,after in zip(old,new):
            self.assertEqual(before['episode_id'],after['episode_id'])
            for key in ['task','start','history','previous_actions','history_length','targets','actions','valid','supervised']:
                np.testing.assert_array_equal(before[key],after[key])
            expected=before['value']-.02*(1-before['value'])/(1-.97)
            np.testing.assert_allclose(expected,after['returns'],rtol=0,atol=0)
            self.assertFalse({'goal','value','risks','collision'}&set(after))
        for w in new[:2]:w['teacher_ids']=np.zeros((11,4),np.int64)
        require_anonymous(batch(new[:2],'cpu'))

    def test_explicit_scalar_collection_is_consumed(self):
        directory=BASE/'predictive_rules_20261015/counterfactual_anonymous_v5'
        manifest=read(directory/'manifest.json');bank=OutcomeBank(directory,manifest['data']['train_layouts'])
        self.assertGreater(len(bank.keys),0)
        import random
        sample=bank.sample(random.Random(1),4,'cpu');require_anonymous(sample)
        for path in directory.glob('*.npz'):
            if path.name.endswith('.tmp.npz'):continue
            with np.load(path,allow_pickle=False) as data:
                self.assertFalse({'collision','eligible_current','eligible_next','goals','risks'}&set(data.files))
                self.assertEqual(data['scalar_rewards'].shape,(len(data['actions']),17))


if __name__=='__main__':unittest.main()
