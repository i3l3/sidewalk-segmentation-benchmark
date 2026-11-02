import unittest
import numpy as np
from PIL import Image,ImageDraw
from src.b2_direction import direction_candidate,_line_pixels

class RouteTests(unittest.TestCase):
    def assert_valid_route(self,mask,result,connected):
        h,w=mask.shape
        path=[(round(y*(h-1)),round(x*(w-1))) for x,y in result['path_xy_normalized']]
        self.assertTrue(path)
        for a,b in zip(path,path[1:]):
            yy,xx=_line_pixels(a,b)
            self.assertTrue(connected[yy,xx].all())
            self.assertTrue((mask[yy,xx]==2).all())

    def test_straight_corridor(self):
        mask=np.ones((120,160),np.uint8);mask[40:,60:100]=2
        result,connected=direction_candidate(mask)
        self.assertEqual(result['direction'],'Straight')
        self.assert_valid_route(mask,result,connected)
        self.assertIsNone(result['features']['curb_proximity'])

    def test_left_bending_corridor(self):
        image=Image.new('L',(160,120),1)
        ImageDraw.Draw(image).polygon([(70,119),(105,119),(48,45),(8,45)],fill=2)
        mask=np.asarray(image);result,connected=direction_candidate(mask)
        self.assertEqual(result['direction'],'Left')
        self.assert_valid_route(mask,result,connected)

    def test_disconnected_component_abstains(self):
        mask=np.ones((120,160),np.uint8);mask[40:90,50:110]=2
        result,connected=direction_candidate(mask)
        self.assertEqual(result['direction'],'No clear path')
        self.assertFalse(connected.any());self.assertFalse(result['path_xy_normalized'])

    def test_road_bike_and_external_obstacle_are_excluded(self):
        mask=np.ones((120,160),np.uint8);mask[40:,60:100]=2;mask[:,20:40]=8
        obstacles=np.zeros_like(mask,bool);obstacles[85:90,60:100]=True
        result,connected=direction_candidate(mask,obstacles)
        self.assertEqual(result['direction'],'No clear path')
        self.assertFalse((connected&obstacles).any())
        self.assertFalse((connected&(mask==8)).any())

    def test_one_pixel_break_does_not_get_bridged(self):
        mask=np.ones((120,160),np.uint8);mask[40:,60:100]=2;mask[90]=1
        result,_=direction_candidate(mask)
        self.assertEqual(result['direction'],'No clear path')
        self.assertFalse(result['path_xy_normalized'])

if __name__=='__main__':unittest.main()
