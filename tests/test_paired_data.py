import hashlib,json,tempfile,unittest
from pathlib import Path
from nesy_vla.paired_data import load_paired_demonstrations


class PairedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.origin={'episode_id':'original','scene':'train_room','start_position':[0,0,0],'start_rotation':[0,0,0,1]}
        self.hashes={'train.json':'train_sha','fixed80.json':'fixed_sha'}
        self.provenance={'source_hashes':self.hashes,'train_layouts':['train_room']}
        self.branches=[]
        for category in ['chair','table']:
            ep=dict(self.origin,episode_id='paired_'+category,category=category)
            file=self.root/(category+'.png');file.write_bytes(b'identical image pixels')
            frame={'episode_id':ep['episode_id'],'category':category,'action':0,'image':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
            self.branches.append({'episode':ep,'frames':[frame],'success':True})
        self.pairs=[{'origin':'original','same_initial_rgb':True,'both_successful':True,'branches':self.branches}]
        self.write()
    def tearDown(self):self.temp.cleanup()
    def write(self):
        values={'source':self.provenance,'pairs':self.pairs,'episodes':[b['episode'] for b in self.branches],
            'examples':[f for b in self.branches for f in b['frames']]}
        for name,value in values.items():(self.root/(name+'.json')).write_text(json.dumps(value))
        complete={'provenance':self.provenance}
        for name in ['pairs','episodes','examples']:complete[name+'_sha256']=hashlib.sha256((self.root/(name+'.json')).read_bytes()).hexdigest()
        (self.root/'COMPLETE').write_text(json.dumps(complete))
    def load(self):return load_paired_demonstrations(self.root,{'original':self.origin},{'train_room'},self.hashes)
    def test_valid_preserves_origins(self):
        extra,rows,proof=self.load();self.assertEqual(len(extra),2);self.assertEqual(proof['pairs'],1)
        self.assertTrue(all(r['paired_goal_demonstration'] for r in rows));self.assertEqual(self.origin['episode_id'],'original')
    def test_reject_heldout_scene(self):
        self.origin['scene']='fixed_room'
        with self.assertRaisesRegex(ValueError,'outside training'):self.load()
    def test_reject_changed_pose(self):
        self.branches[0]['episode']['start_position']=[1,0,0];self.write()
        with self.assertRaisesRegex(ValueError,'start differs'):self.load()
    def test_reject_bad_image(self):
        (self.root/'chair.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):self.load()
    def test_reject_unmatched_images(self):
        (self.root/'chair.png').write_bytes(b'changed');self.branches[0]['frames'][0]['sha256']=hashlib.sha256(b'changed').hexdigest();self.write()
        with self.assertRaisesRegex(ValueError,'equality'):self.load()
    def test_reject_unsuccessful(self):
        self.branches[0]['success']=False;self.write()
        with self.assertRaisesRegex(ValueError,'Unsuccessful'):self.load()
    def test_reject_manifest_tamper(self):
        (self.root/'examples.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'checksum'):self.load()
    def test_reject_split_hash(self):
        self.hashes=dict(self.hashes,**{'fixed80.json':'other'})
        with self.assertRaisesRegex(ValueError,'source split'):self.load()


if __name__=='__main__':unittest.main()
