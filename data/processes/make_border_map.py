import warnings
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from concern.config import State
from .data_process import DataProcess
from data.data_utils import generate_rbox
from charnet.modeling.postprocessing import load_char_rev_dict
import yaml


class MakeBorderMap(DataProcess):
    r'''
    Making the border map from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    shrink_ratio = State(default=0.4)
    thresh_min = State(default=0.3)
    thresh_max = State(default=0.7)
    charindx = State(default="")
    yamlfile = State(default="")
    char_rev_dict = []
    

    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.debug = False
        warnings.simplefilter("ignore")
        if (self.charindx !=""):
            self.char_rev_dict=load_char_rev_dict(self.charindx)
        if self.yamlfile !="":
            self.yamlf=open(self.yamlfile, 'w')
                   
            
    def process(self, data, *args, **kwargs):
        r'''
        required keys:
            image, polygons, ignore_tags
        adding keys:
            thresh_map, thresh_mask
        '''
        image = data['image']
        polygons = data['polygons']
        lines_text = data['lines_text']
        ignore_tags = data['ignore_tags']
        #canvas = np.zeros(image.shape[:2], dtype=np.float32)
        #mask = np.zeros(image.shape[:2], dtype=np.float32)

        #for i in range(len(polygons)):
        #    if ignore_tags[i]:
        #        continue
        #    self.draw_border_map(polygons[i], canvas, mask=mask)
        # imsize = (h,w), image.shape[0]=h, image.shape[1]=>w
        score_map, geo_map, training_mask, rects=generate_rbox((data['image'].shape[0], data['image'].shape[1]), polygons, ignore_tags, lines_text)
        if (self.debug == True):
            for idx, (po, rec) in enumerate(zip(polygons, rects)):
                print('Poly:', po)
                print('Rect', rec['rect'])
            
        data['score_map'] = score_map
        data['geo_map'] = geo_map
        data['training_mask'] = training_mask
        
        #canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        #data['thresh_map'] = canvas
        #data['thresh_mask'] = mask
        
        polygons_char = data['polygons_char']
        ignore_tags_char = data['ignore_tags_char']
        lines_char = data['lines_char']
        
        
        
        #canvas_char = np.zeros(image.shape[:2], dtype=np.float32)
        #mask_char = np.zeros(image.shape[:2], dtype=np.float32)

        #for i in range(len(polygons_char)):
        #    if ignore_tags_char[i]:
        #        continue
        #    self.draw_border_map(polygons_char[i], canvas_char, mask=mask_char)
            
        score_map_char, geo_map_char, training_mask_char, rects_char=generate_rbox((data['image'].shape[0], data['image'].shape[1]), polygons_char, ignore_tags_char, lines_char, 'C', self.char_rev_dict)
        data['score_map_char'] = score_map_char         # Character detection
        data['geo_map_char'] = geo_map_char             # Character recognization 
        data['training_mask_char'] = training_mask_char
        #print('Rects_Char', rects_char)
        #print('Poly_Char:', polygons_char)

        #canvas_char = canvas_char * (self.thresh_max - self.thresh_min) + self.thresh_min
        #data['thresh_map_char'] = canvas_char
        #data['thresh_mask_char'] = mask_char        
        
        
        
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2

