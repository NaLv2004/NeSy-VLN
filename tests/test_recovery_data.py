import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from nesy_vla.recovery_data import load_recovery_demonstrations
from nesy_vla.resources import recovery_demonstration_schedule


class RecoveryDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.images=self.root/'images';self.images.mkdir()
        self.episode={'episode_id':'source_episode','scene':'train_scene','category':'object','start_position':[0,0,0]}
        self.hashes={'train.json':'lockedtrain','fixed80.json':'lockedfixed'}
        self.source={'source_hashes':self.hashes,'train_layouts':['train_scene'],'image_root':str(self.images)}
        frames=[]
        for i,(action,mask) in enumerate([(3,False),(2,True),(0,True)]):
            path=self.images/f'{i:03d}.png';path.write_bytes(bytes([i]))
            frames.append(dict(episode_id='source_episode',category='object',action=action,supervision_mask=mask,
                image=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),public_feedback={'collision':i==0}))
        self.records=[dict(episode=copy.deepcopy(self.episode),prefix_actions=[3],success=True,frames=frames)];self.write()
    def tearDown(self):self.temp.cleanup()
    def write(self):
        for name,value in [('source',self.source),('recoveries',self.records),('examples',[f for r in self.records for f in r['frames']])]:
            (self.root/(name+'.json')).write_text(json.dumps(value))
        complete={'provenance':self.source}
        for name in ['recoveries','examples']:complete[name+'_sha256']=hashlib.sha256((self.root/(name+'.json')).read_bytes()).hexdigest()
        (self.root/'COMPLETE').write_text(json.dumps(complete))
    def load(self):return load_recovery_demonstrations(self.root,{'source_episode':self.episode},{'train_scene'},self.hashes,self.images)
    def test_prefix_is_context_and_feedback_survives(self):
        ids,rows,evidence=self.load();self.assertEqual(evidence['context_frames'],1)
        self.assertEqual([r['action'] for r in rows],[3,2,0]);self.assertEqual([r['supervision_mask'] for r in rows],[False,True,True])
        self.assertTrue(rows[0]['public_feedback']['collision'])
    def test_prefix_must_not_be_supervised(self):
        self.records[0]['frames'][0]['supervision_mask']=True;self.write()
        with self.assertRaisesRegex(ValueError,'prefix'):self.load()
    def test_reject_development_or_fixed_scene(self):
        self.records[0]['episode']['scene']='heldout';self.episode['scene']='heldout';self.write()
        with self.assertRaisesRegex(ValueError,'training split'):self.load()
    def test_reject_pose_or_category_change(self):
        self.records[0]['episode']['start_position']=[1,0,0];self.write()
        with self.assertRaisesRegex(ValueError,'training split'):self.load()
    def test_reject_changed_images(self):
        (self.images/'001.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):self.load()
    def test_reject_unsuccessful_or_long_trajectories(self):
        self.records[0]['success']=False;self.write()
        with self.assertRaisesRegex(ValueError,'successfully'):self.load()
    def test_mixed_schedule_is_disjoint_balanced_and_causal_pairs_stay_together(self):
        base=[f'base{i}' for i in range(200)];pairs=[(f'a{i}',f'b{i}') for i in range(100)];recovery=[f'recover{i}' for i in range(100)]
        outputs=[recovery_demonstration_schedule(base,pairs,recovery,1,r,3,24,48,.25,.25) for r in range(3)]
        self.assertEqual(len(set(x for row in outputs for x in row)),72)
        for row in outputs:
            self.assertEqual(sum(x in base for x in row),12);self.assertEqual(sum(x in recovery for x in row),6)
            self.assertEqual(sum(set(pair).issubset(row) for pair in pairs),3)


if __name__=='__main__':unittest.main()
