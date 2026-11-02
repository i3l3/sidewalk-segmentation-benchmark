from pathlib import Path
import io
import json
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image
from src.schema import *
from src.dataset import *
from src.labeling import import_mask,import_review_batch
from src.evaluation import confusion,metrics,boundary_counts,boundary_f1
from src.direction import extract_features,rule_direction

class PipelineTests(unittest.TestCase):
    def fixture(self,folder):
        root=init(Path(folder)/'project');source=Path(folder)/'input';source.mkdir()
        rng=np.random.default_rng(4)
        for i in range(9):Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(source/f'{i}.png')
        data=import_images(root,source)
        for i,r in enumerate(data):r['group_id']='place_'+str(i)
        write_rows(root,data);split_dataset(root,targets=(5,2,2),confirm_groups=True)
        return root

    def test_mapping_and_ignore(self):
        x=np.array([[0,1,2,3,4,5,6,7,8,9,10,11,65535]],np.uint16)
        self.assertEqual(legacy_to_train(x).tolist(),[[0,1,2,3,4,255,255,5,6,7,8,255,255]])

    def test_group_confirmation_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=init(temp)
            with self.assertRaises(ValueError):split_dataset(root)

    def test_split_lock_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(temp);data=rows(root);data[0]['split']='test' if data[0]['split']!='test' else 'train';write_rows(root,data)
            with self.assertRaises(ValueError):validate_frozen(root)

    def test_review_requires_person_and_invalidates_changed_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(temp);row=rows(root)[0];p=Path(temp)/'label.png'
            Image.fromarray(np.full((32,32),2,np.uint8)).save(p)
            with self.assertRaises(ValueError):import_mask(root,row['image_id'],p,verified=True)
            r=import_mask(root,row['image_id'],p,'tester',True)
            Image.fromarray(np.full((32,32),1,np.uint8)).save(root/r['mask_path'])
            refreshed=refresh_reviews(root)
            self.assertEqual(next(r for r in refreshed if r['image_id']==row['image_id'])['label_verified'],'false')

    def test_rgb_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(temp);r=rows(root)[0];p=Path(temp)/'rgb.png';Image.new('RGB',(32,32),'red').save(p)
            with self.assertRaises(ValueError):import_mask(root,r['image_id'],p)

    def test_unknown_prediction_is_error(self):
        cm=confusion(np.array([[2,2,255]]),np.array([[2,255,1]]));m=metrics(cm)
        self.assertEqual(cm.sum(),2);self.assertEqual(m['per_class'][2]['iou'],.5)

    def test_unsupported_na(self):
        m=metrics(confusion(np.array([[3,2]]),np.array([[0,2]])),[0,1,2])
        self.assertIsNone(m['per_class'][3]['iou'])

    def test_boundary_perfect(self):
        m=np.zeros((30,30),np.uint8);m[8:22,8:22]=2
        self.assertEqual(boundary_f1(boundary_counts(m,m,2)),1.)

    def test_disconnected_and_center_path(self):
        m=np.zeros((100,100),np.uint8);m[10:60,30:70]=2
        f,c=extract_features(m);self.assertFalse(c.any());self.assertEqual(rule_direction(f),'No clear path')
        m[:,30:70]=2;f,c=extract_features(m);self.assertEqual(rule_direction(f),'Straight')
        self.assertIsNone(f['obstacle_ratio'])

    def test_zip_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../outside.txt','bad')
            with self.assertRaises(ValueError):extract_zip(p,Path(temp)/'target')

    def test_duplicates_and_exploratory_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=init(Path(temp)/'project');source=Path(temp)/'input';source.mkdir()
            a=Image.new('RGB',(32,32),'red');a.save(source/'a.png');a.save(source/'b.png')
            data=import_images(root,source);self.assertEqual(len(data),1)
            other=init(Path(temp)/'other');save_json(other/'data/excluded_hashes.json',[data[0]['pixel_sha']])
            self.assertEqual(import_images(other,source),[])

    def test_teacher_selection_never_uses_test_labels(self):
        from src.labeling import select_teacher,TEACHERS
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(temp);data=rows(root)
            for i,r in enumerate(data):
                truth=1 if r['split']=='test' else 2
                path=Path(temp)/f'm{i}.png';Image.fromarray(np.full((32,32),truth,np.uint8)).save(path)
                import_mask(root,r['image_id'],path,'synthetic fixture',True)
                for key,c in zip(TEACHERS,[2,1]):
                    p=root/'data/pseudo_labels'/key/f'{r["image_id"]}.png';p.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.full((32,32),c,np.uint8)).save(p)
            result=select_teacher(root)
            self.assertEqual(result['selected'],TEACHERS[0])
            self.assertTrue(all(r['split']!='test' for r in result['reviewed_selection_samples']))

    def test_similar_images_stay_together_even_with_different_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root=init(Path(temp)/'project');source=Path(temp)/'input';source.mkdir()
            rng=np.random.default_rng(54)
            for i in range(9):Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(source/f'{i}.png')
            data=import_images(root,source)
            for i,r in enumerate(data):r['group_id']='place_'+str(i)
            data[1]['dhash']=data[0]['dhash'];write_rows(root,data)
            split_dataset(root,confirm_groups=True)
            current={r['image_id']:r['split'] for r in rows(root)}
            self.assertEqual(current[data[0]['image_id']],current[data[1]['image_id']])

    def direction_fixture(self,counts):
        import pandas as pd
        from src.direction import FEATURES,DIRECTIONS
        records=[]
        for split,n in counts.items():
            for c,name in enumerate(DIRECTIONS):
                for i in range(n):
                    records.append({'image_id':f'{split}_{c}_{i}','model':'synthetic','split':split,
                        'group_id':f'{split}_{c}_{i}','direction_verified':'true','direction_label':name,
                        'has_seed_connection':c!=3,**dict(zip(FEATURES,[.8 if c==0 else .1,.8 if c==1 else .1,.8 if c==2 else .1,.9 if c!=3 else 0,.3,0.,.2]))})
        return pd.DataFrame(records)

    def test_direction_classifier_declines_insufficient_labels(self):
        from src.direction import classify
        with tempfile.TemporaryDirectory(prefix='synthetic') as temp:
            root=init(temp);classify(root,self.direction_fixture({'train':2,'validation':1,'test':1}),'synthetic')
            status=load_json(root/'metrics/direction/synthetic/status.json')
            self.assertFalse(status['trained']);self.assertFalse(list(root.rglob('*.joblib')))
            self.assertTrue((root/'metrics/direction/synthetic/rule_metrics.json').exists())

    def test_direction_classifier_trains_only_when_counts_sufficient(self):
        from src.direction import classify
        with tempfile.TemporaryDirectory(prefix='synthetic') as temp:
            root=init(temp);classify(root,self.direction_fixture({'train':20,'validation':5,'test':5}),'synthetic')
            self.assertTrue(load_json(root/'metrics/direction/synthetic/status.json')['trained'])
            self.assertTrue((root/'metrics/direction/synthetic/decision_tree.joblib').exists())
            self.assertTrue((root/'metrics/direction/synthetic/random_forest_confusion.png').exists())

if __name__=='__main__':unittest.main(verbosity=2)
