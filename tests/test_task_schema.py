import tempfile,unittest
from pathlib import Path
import numpy as np
from src.schema import configure_task,active_class_ids,encode_task_mask,decode_task_mask,restrict_task_mask,save_json
from src.evaluation import metrics,confusion

class TaskTests(unittest.TestCase):
    def test_dense_head_roundtrip_preserves_ignore(self):
        a=np.array([[0,1,2,3,8,255]],np.uint8)
        encoded=encode_task_mask(a,[1,2,8])
        self.assertEqual(encoded.tolist(),[[255,0,1,255,2,255]])
        self.assertEqual(decode_task_mask(encoded,[1,2,8]).tolist(),[[255,1,2,255,8,255]])
        with self.assertRaises(ValueError):decode_task_mask(np.array([[3]]),[1,2,8])

    def test_task_is_separate_from_canonical_mask_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure_task(tmp,[1,2,8]);self.assertEqual(active_class_ids(tmp),[1,2,8])
            save_json(Path(tmp)/'data/splits/locked.json',{})
            configure_task(tmp,[1,2,8])
            with self.assertRaises(ValueError):configure_task(tmp,list(range(9)))

    def test_metrics_only_score_active_classes(self):
        gt=restrict_task_mask(np.array([[1,2,8,3]],np.uint8),[1,2,8])
        pred=np.array([[1,2,1,3]],np.uint8)
        m=metrics(confusion(gt,pred),[1,2,8])
        self.assertAlmostEqual(m['miou'],.5)
        self.assertIsNone(m['per_class'][3]['iou'])
        self.assertAlmostEqual(m['pixel_accuracy'],2/3)

if __name__=='__main__':unittest.main()
