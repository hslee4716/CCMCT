# ReadableDicomDataset 객체 - DCM 파일 조작을 위한 

import numpy as np
import pandas as pd
from pydicom import dataset

from pydicom.encaps import decode_data_sequence
from PIL import Image
from typing import Union
from glob import glob
import io
import pydicom
import os
import sqlite3


class ReadableDicomDataset():
    def __init__(self, filename):
        self._ds = pydicom.dcmread(filename)
        self.geometry_imsize = (self._ds[0x48,0x6].value,self._ds[0x48,0x7].value)
        self.geometry_tilesize = (self._ds.Columns, self._ds.Rows)
        self.geometry_columns = round(0.5+(self.geometry_imsize[0]/self.geometry_tilesize[0]))
        self.geometry_rows = round(0.5 + (self.geometry_imsize[1] / self.geometry_tilesize[1] ))
        self._dsequence = decode_data_sequence(self._ds.PixelData)

    def imagePos_to_id(self, imagePos:tuple):
        id_x, id_y = imagePos
        return (id_x+(id_y*self.geometry_columns))
    
    def get_tile(self, pos):
        return np.array(Image.open(io.BytesIO(self._dsequence[pos])))
        

    def get_id(self, pixelX:int, pixelY:int) -> Union[int, int, int]:

        id_x = round(-0.5+(pixelX/self.geometry_tilesize[1]))
        id_y = round(-0.5+(pixelY/self.geometry_tilesize[0]))

        return (id_x,id_y), pixelX-(id_x*self.geometry_tilesize[0]), pixelY-(id_y*self.geometry_tilesize[1]),

    @property
    def dimensions(self):
        return self.geometry_imsize
        
    def read_region(self, location: tuple, size:tuple):
        lu, lu_xo, lu_yo = self.get_id(*list(location))
        rl, rl_xo, rl_yo = self.get_id(*[sum(x) for x in zip(location,size)])
        # generate big image
        bigimg = np.zeros(((rl[1]-lu[1]+1)*self.geometry_tilesize[0], (rl[0]-lu[0]+1)*self.geometry_tilesize[1], self._ds[0x0028, 0x0002].value), np.uint8)
        for xi, xgridc in enumerate(range(lu[0],rl[0]+1)):
            for yi, ygridc in enumerate(range(lu[1],rl[1]+1)):
                if (xgridc<0) or (ygridc<0):
                    continue
                bigimg[yi*self.geometry_tilesize[0]:(yi+1)*self.geometry_tilesize[0],
                       xi*self.geometry_tilesize[1]:(xi+1)*self.geometry_tilesize[1]] = \
                       self.get_tile(self.imagePos_to_id((xgridc,ygridc)))
        # crop big image
        return bigimg[lu_yo:lu_yo+size[1],lu_xo:lu_xo+size[0]]


def fix_labels(image_dir, label_dir):
    file_lists = glob(label_dir + "/*.txt")

    remove_count = 0
    fix_count = 0
    
    count = -1
    for text in file_lists:
        count += 1
        temp = text.split("\\")[1].split(".")[0]
        txt_file = label_dir + "/" + temp + ".txt"
        png_file = image_dir + "/" + temp + ".png"
        if count > 10000:
            break
        with open(txt_file, 'r') as f:
            cells = f.readlines()
            fixed_cells = []

            for cell in cells:
                temp_cell = cell.strip('\n').split()
                if float(temp_cell[1]) + float(temp_cell[3])/2 > 1 or float(temp_cell[1]) - float(temp_cell[3])/2 < 0 or \
                            float(temp_cell[2]) + float(temp_cell[4])/2 > 1 or float(temp_cell[2]) - float(temp_cell[4])/2 < 0 :
                    temp_cell = None
                    fix_count += 1
                if temp_cell != None:
                    fixed_cells.append(temp_cell)
        
        with open(txt_file, 'w') as f:
            for fixed_cell in fixed_cells:
                f.write((' ').join(fixed_cell) + '\n')

        if len(fixed_cells) == 0 or len(cells) == 0:
            f.close()
            os.remove(txt_file)
            os.remove(png_file)
            remove_count += 1
    
    return remove_count, fix_count


    
def dcm_to_train_set(db = "datasets/archive/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite", source_dir = "datasets/archive",\
                         dest_dir = "datasets/train", tile_size = 1000, cell_size = 40):
        
    DB = sqlite3.connect(db)
    cur = DB.cursor()

    IMGSZ = tile_size

    datasets = glob(source_dir +"/*.dcm")

    file_names = []
    folder_names = []
    for data in datasets:
        temp = data.split("\\")[1]
        file_names.append(temp)
        folder_names.append(temp.split('.')[0])

    folder_names.sort()
    file_names.sort()

    for folder_name, file_name in zip(folder_names, file_names):
        print(folder_name)
        
        # filename -> slide 번호, width, height 추출
        slide = cur.execute(f"""SELECT uid, width, height
                                from Slides 
                                where filename == "{file_name}" """).fetchall()
        slide = slide[0]

        save_dir = dest_dir + '/'
        
        # 이미지 읽기 위한 Dicom Dataset 객체 객체 생성
        ds = ReadableDicomDataset(source_dir + file_name)
        idx = -1
        for width in range(0,slide[1]+IMGSZ,IMGSZ):
            for height in range(0,slide[2]+IMGSZ,IMGSZ):
                idx += 1
                location = (width, height)
                size = (IMGSZ,IMGSZ)
                # cells = (현재 location에서의 x, y, uid)
                cells = cur.execute(f"""SELECT coordinateX-{location[0]}, coordinateY-{location[1]}, annoId
                                from Annotations_coordinates where slide=={slide[0]} and 
                                coordinateX>{location[0]} and coordinateX<{location[0]+size[0]} and 
                                coordinateY>{location[1]} and coordinateY<{location[1]+size[1]}""").fetchall()
                if len(cells) > 0:
                    img = Image.fromarray(ds.read_region(location=location,size=size))
                    img.save(save_dir + "images/" + folder_name + "_" + str(idx) + ".png", 'png')
                    
                    file = save_dir + "labels/" + folder_name + "_" + str(idx) + ".txt"
                    if os.path.isfile(file):
                        os.remove(file)

                    with open(file, 'w') as f:
                        for cell in cells:
                            # Annotations_coordinates에는 존재하지만 해당 annoid가 Annotations에 존재하지 않는 경우 저장하지 않음.
                            try:
                                cell_class = cur.execute(f"""SELECT agreedClass FROM Annotations where uid == {cell[2]}""").fetchall()[0][0]
                                label, x, y, w, h = cell_class, cell[0]/IMGSZ, cell[1]/IMGSZ, cell_size/IMGSZ, cell_size/IMGSZ
                                
                                line = str(label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
                                f.write(line)
                            except:
                                None

def get_all_cells(label_dir):
    target_dir = label_dir+"/"
    txt_list = glob(target_dir + "*.txt")
    cells = []
    for txt in txt_list:
        file = txt.split("\\")[1]
        file = target_dir + file

        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.strip().split()
                cells.append(temp)

    cells = pd.DataFrame(cells, columns=["label", "x", "y", "w", "h"])
    return cells
