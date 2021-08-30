import json
import pandas as pd
from pandas.core.indexes import category
from sklearn.model_selection import train_test_split
import numpy as np

temp_dict = {
    "car": {"index": [1,2,3,4,5]},
    "horse" : 100
}

with open('test.json', 'w', encoding='utf-8') as make_file:
    json.dump(temp_dict, make_file, indent="\t")


class coco():
    def __init__(self):
        self.dict_all = {
            "info":{
                "description": None,
                "url": None,
                "version": None,
                "year": None,
                "contributor": None,
                "date_created": None
            },
            "licenses":[
                {
                    "url":None,
                    "id":None,
                    "name":None
                }],
            "images":
            [
                {
                    "file_name": None,
                    "height": None,
                    "width": None,
                    "date_captured": None,
                    "flickr_url": None,
                    "id": None
                }
            ],
            "annotations":{},
            "categories":{}
        }
    
    def coco_makeAnnotation(cells:pd.DataFrame, imgid, annoid, cellsize):
        annos = []
        for cell in cells:
            anno = {
                "segmentation": [],
                "area": cellsize*cellsize,
                "iscrowd": 0,
                "image_id": imgid,
                "bbox": {cell[0]-cellsize//2, cell[1]-cellsize//2, cellsize, cellsize},
                "category_id": cell[2],
                "id": annoid
            }
            annos.append(anno)
        return annos
    def coco_makeImage(file_name, width, height, id):
        img = {
            "file_name": file_name,\
            "height": height,\
            "width": width,\
            "id": id
        }
        return img
    def make_categories(LABELS):
        categories = []
        for idx,label in enumerate(LABELS):
            category = {
                "supercategory": label,
                "id": idx,
                "name": label
            }
            categories.append(category)


a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

b, c = train_test_split(a, test_size=0.3)
print(b, c)
