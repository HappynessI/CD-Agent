import unittest
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from render import crop_box,SIZE
from agent_baseline.analysis.t3_routes_match import bbox
from agent_baseline.analysis.night_t3_local_raw import local_raw_views


class RawViewsTests(unittest.TestCase):
    def test_same_crop_raw_pixels_and_thin_hole_edges(self):
        image=np.full((96,96,3),120,np.uint8);mask=np.zeros((96,96),bool)
        mask[30:60,30:60]=True;mask[40:50,40:50]=False
        output=local_raw_views([image,image+20],mask);self.assertEqual(len(output),2)
        l,t,r,b=crop_box(bbox(mask),mask.shape,scale=1.5,min_side=64)
        support=np.asarray(Image.fromarray(mask[t:b,l:r]).resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
        boundary=support&~ndi.binary_erosion(support,border_value=0)
        for i,(_,rendered) in enumerate(output):
            array=np.asarray(rendered)
            self.assertTrue((array[~boundary]==120+20*i).all())
            self.assertTrue((array[boundary]==[255,0,0]).all())
        self.assertTrue((image==120).all())


if __name__=='__main__':unittest.main()
