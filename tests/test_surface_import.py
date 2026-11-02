from pathlib import Path
import io,json,tempfile,unittest,zipfile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from src.surface_import import rasterize,prepare_surface
from src.dataset import rows

class SurfaceTests(unittest.TestCase):
    def test_ignore_overlay_zorder_and_unannotated(self):
        element=ET.fromstring('<image width="10" height="10"><polygon label="alley" z_order="2" points="2,2;7,2;7,7;2,7"/><polygon label="sidewalk" z_order="1" points="0,0;8,0;8,8;0,8"/></image>')
        train,native=rasterize(element,{'alley':0,'sidewalk':1})
        self.assertEqual(train[1,1],2);self.assertEqual(train[3,3],255);self.assertEqual(train[9,9],255)
        self.assertEqual(native[3,3],0)

    def test_roadway_crosswalk_does_not_become_road(self):
        element=ET.fromstring('<image width="4" height="4"><polygon label="roadway" points="0,0;3,0;3,3;0,3"><attribute name="attribute">crosswalk</attribute></polygon></image>')
        train,_=rasterize(element,{'roadway':0});self.assertTrue((train==255).all())
        element[0][0].text='normal';train,_=rasterize(element,{'roadway':0});self.assertTrue((train==1).all())

    def test_unsupported_shape_refuses_partial_mask(self):
        with self.assertRaises(ValueError):rasterize(ET.fromstring('<image width="4" height="4"><polyline label="sidewalk" points="0,0;1,1"/></image>'),{'sidewalk':0})

    def test_prepare_resume_preserves_review_and_native(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'surface.zip';root=Path(temp)/'project'
            with zipfile.ZipFile(source,'w') as archive:
                for group in range(2):
                    folder=f'Surface_{group:03d}';images=[]
                    for frame in range(3):
                        name=f'frame{frame}.jpg';buffer=io.BytesIO()
                        Image.fromarray(np.full((12,12,3),20+group*20+frame,np.uint8)).save(buffer,format='JPEG')
                        archive.writestr(folder+'/'+name,buffer.getvalue());buffer=io.BytesIO()
                        Image.fromarray(np.zeros((12,12,3),np.uint8)).save(buffer,format='PNG')
                        archive.writestr(folder+'/MASK/'+Path(name).stem+'.png',buffer.getvalue())
                        images.append(f'<image name="{name}" width="12" height="12"><polygon label="sidewalk" z_order="0" points="0,0;11,0;11,11;0,11"/></image>')
                    archive.writestr(folder+'/labels.xml','<annotations><meta><task><name>sequence</name><labels><label><name>sidewalk</name></label></labels></task></meta>'+''.join(images)+'</annotations>')
            report=prepare_surface(root,source,count=5)
            self.assertEqual(report['prepared_count'],5);self.assertFalse(report['review_complete'])
            data=rows(root);self.assertEqual(len(data),5)
            self.assertTrue(all(r['label_verified']=='false' and r['status']=='needs_review' and r['group_id']=='' and r['split']=='' for r in data))
            self.assertEqual(len(list((root/'data/native_labels').glob('*.png'))),5)
            self.assertEqual(len(list((root/'data/source_rgb_masks').glob('*.png'))),5)
            before=[r['image_sha'] for r in data]
            report=prepare_surface(root,source,count=5)
            self.assertEqual(before,[r['image_sha'] for r in rows(root)])
            with self.assertRaises(ValueError):prepare_surface(root,source,count=4)

if __name__=='__main__':unittest.main()
