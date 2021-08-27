# ReadableDicomDataset 객체 - DCM 파일 조작을 위한 

import numpy as np
import pandas as pd
from pydicom import dataset
from sklearn.model_selection import train_test_split
import shutil

from pydicom.encaps import decode_data_sequence
from PIL import Image
from typing import Union
from glob import glob
import io
import pydicom
import os
import sqlite3
import time



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


def yolo_fix_labels(image_dir, label_dir):
    file_lists = glob(label_dir + "/*.txt")

    for text in file_lists:
        temp = text.split("\\")[1].split(".")[0]
        txt_file = label_dir + "/" + temp + ".txt"
        png_file = image_dir + "/" + temp + ".png"

        with open(txt_file, 'r') as f:
            cells = f.readlines()
            fixed_cells = []

            for cell in cells:
                temp_cell = cell.strip().split()
                print(temp_cell)

                if '7' in temp_cell[0]:
                    temp_cell[0] = '0'
                
                if '4' in temp_cell[0]:
                    temp_cell = None

                if float(temp_cell[1]) + float(temp_cell[3])/2 > 1 or float(temp_cell[1]) - float(temp_cell[3])/2 < 0 or \
                            float(temp_cell[2]) + float(temp_cell[4])/2 > 1 or float(temp_cell[2]) - float(temp_cell[4])/2 < 0 :
                    temp_cell = None

                if temp_cell != None:
                    fixed_cells.append(temp_cell)
        
        with open(txt_file, 'w') as f:
            for fixed_cell in fixed_cells:
                f.write((' ').join(fixed_cell) + '\n')

        if len(fixed_cells) == 0 or len(cells) == 0:
            os.remove(txt_file)
            os.remove(png_file)




def yolo_dcm_to_train_set_includeNone(db = "original_data/archive/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite", source_dir = "original_data/archive",\
                         dest_dir = "datasets/origin", tile_size = 320, cell_size = 40, LABELS = ["Mitotic_figure_lookalike", "granulocyte", "mitotic_figure", "tumor_cell"]):


    if not os.path.isdir(dest_dir):                                                           
        os.mkdir(dest_dir)
        os.mkdir(dest_dir+"/images")
        os.mkdir(dest_dir+"/labels")

    DB = sqlite3.connect(db)
    cur = DB.cursor()
    IMGSZ = tile_size
    datasets = glob(source_dir +"/*.dcm")

    file_names = []
    for data in datasets:
        temp = data.split("\\")[1]
        file_names.append(temp.split('.')[0])

    file_names.sort()
    file_len = len(file_names)
    file_count = 0

    for file_name in  file_names:
        file_count += 1
        # filename -> slide 번호, width, height 추출
        slide = cur.execute(f"""SELECT uid, width, height from Slides where filename == "{file_name+".dcm"}" """).fetchall()
        slide = slide[0]
        
        # 이미지 읽기 위한 Dicom Dataset 객체 객체 생성
        ds = ReadableDicomDataset(source_dir+ "/" + file_name + ".dcm")
        tiles = ds.geometry_imsize
        tiles = int(tiles[0]/IMGSZ) * int(tiles[1]/IMGSZ)

        Annotations = pd.DataFrame(cur.execute(f"""SELECT uid, agreedClass FROM Annotations WHERE slide == {slide[0]}""").fetchall(),\
                                    columns=["annoid", "class"]).set_index('annoid')
       
        cells = pd.DataFrame(cur.execute(f"""SELECT coordinateX, coordinateY, annoId
                                from Annotations_coordinates where slide=={slide[0]}""").fetchall(),\
                                    columns=['x', 'y', 'annoid'])

        idx = -1
        
        for col in range(0,slide[2]-IMGSZ,IMGSZ):
            for row in range(0,slide[1]-IMGSZ,IMGSZ):
                idx += 1

                if idx % 500 == 0:
                    print(file_count, "/", file_len, file_name,":",idx, "/", tiles)

                location = (row, col)
                local_cells = cells[(cells['x'] > location[0])&(cells['y'] > location[1])&\
                            (cells['x']<location[0]+IMGSZ)&(cells['y']<location[1]+IMGSZ)]
                local_cells['x'] -= location[0]
                local_cells['y'] -= location[1]

                lines = []
                flag = 0
                for cell in local_cells.values.tolist():
                    try:
                        label = Annotations.loc[cell[2]].values[0]
                    except:
                        # annoid가 annotations에 존재하지 않는 경우 걸러냄.
                        continue

                    # 원래 x1, y1, x2, y2이지만 편의상 w, h로 표기
                    x, y, w, h = cell[0]/IMGSZ, cell[1]/IMGSZ, cell_size/IMGSZ, cell_size/IMGSZ

                    if label == 7:
                        label = 0

                    #imgsize 범위 벗어나는 경우, label 0 ~ 3이 아닌 경우 pass
                    if x - w/2 <= 0 or x + w/2 >= 1 or  y - h/2 <= 0 or y + h/2 >= 1 or label >= 4:
                                continue
                    
                    line = str(label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
                    lines.append(line)
                    flag = 1

                # 빈 라벨의 경우 1/10만 학습, 아니면 124만장 나옴
                if flag == 1 or idx % 10 == 0:
                    with open(dest_dir+'/labels/'+file_name+'_'+str(idx)+'.txt', 'w') as f:
                        for l in lines:
                            f.write(l)

                    # 이미지 저장
                    img = Image.fromarray(ds.read_region(location=location,size=(IMGSZ,IMGSZ)))
                    img.save(dest_dir + "/images/" + file_name + "_" + str(idx) + ".png", 'png')

# yolo_dcm_to_png_includeNone -> yolo_train_test_split
def yolo_train_test_split(source_dir="datasets/origin", dest_dir = "datasets/New"):
    
    files = glob(source_dir+"/labels/*.txt")
    
    file_lists = []
    for file_name in files:
        file_lists.append(file_name.split("\\")[1].split(".")[0])

    train_files, test_files = train_test_split(file_lists, random_state=7, test_size=0.2)
    print(len(train_files), len(test_files))

    with open(source_dir+"/train.txt", 'w') as f:
        for train_file in train_files:
            f.write(train_file+"\n")

    with open(source_dir+"/val.txt", 'w') as f:
        for test_file in test_files:
            f.write(test_file+"\n")

    # train test split 칸에서 작성된 txt파일에 따라 이미지, 라벨 다른 폴더로 분할

    img_dest = dest_dir + "/images"
    label_dest = dest_dir + "/labels"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(img_dest):
        os.makedirs(img_dest)
        os.makedirs(img_dest + "/train")
        os.makedirs(img_dest + "/val")

    if not os.path.exists(label_dest):
        os.makedirs(label_dest)
        os.makedirs(label_dest + "/train")
        os.makedirs(label_dest + "/val")

    train_txt = source_dir + "/train.txt"
    val_txt = source_dir + "/val.txt"

    with open(train_txt, 'r') as f:
        train_files = f.readlines()

    with open(val_txt, 'r') as f:
        val_files = f.readlines()    

    for train_file in train_files:
        temp = train_file.strip()
        shutil.move(source_dir + "/images/"+temp+".png", img_dest + "/train")
        shutil.move(source_dir + "/labels/"+temp+".txt", label_dest + "/train")

    for val_file in val_files:
        temp = val_file.strip() 
        shutil.move(source_dir + "/images/"+temp+".png", img_dest + "/val")
        shutil.move(source_dir + "/labels/"+temp+".txt", label_dest + "/val")

## M2det expired!
# def m2det_dcm_to_train_set(db = "original_data/archive/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite", source_dir = "original_data/archive",\
#                          dest_dir = "datasets/img_320", tile_size = 640, cell_size = 40, LABELS = ["Mitotic_figure_lookalike", "granulocyte", "mitotic_figure", "tumor_cell"]):

#     DB = sqlite3.connect(db)
#     cur = DB.cursor()
#     IMGSZ = tile_size
#     datasets = glob(source_dir +"/*.dcm")

#     if not os.path.isdir(dest_dir):                                                           
#         os.mkdir(dest_dir)
        
#     if not os.path.isdir(dest_dir+"/images"):                                                           
#         os.mkdir(dest_dir+"/images")

#     file_names = []
#     for data in datasets:
#         temp = data.split("\\")[1]
#         file_names.append(temp.split('.')[0])

#     file_names.sort()
#     file_len = len(file_names)
#     file_count = 0
#     with open(dest_dir+'/'+'labels.csv','w') as f:
#         for index,l in enumerate(LABELS):
#             f.write(l+','+str(index)+'\n')


#     f = open(dest_dir+"/"+"train.csv", 'w')

#     for file_name in  file_names:
#         file_count += 1
#         # filename -> slide 번호, width, height 추출
#         slide = cur.execute(f"""SELECT uid, width, height from Slides where filename == "{file_name+".dcm"}" """).fetchall()
#         slide = slide[0]
        
#         # 이미지 읽기 위한 Dicom Dataset 객체 객체 생성
#         ds = ReadableDicomDataset(source_dir+ "/" + file_name + ".dcm")
#         tiles = ds.geometry_imsize
#         tiles = int(tiles[0]/IMGSZ) * int(tiles[1]/IMGSZ)

#         Annotations = pd.DataFrame(cur.execute(f"""SELECT uid, agreedClass FROM Annotations WHERE slide == {slide[0]}""").fetchall(),\
#                                     columns=["annoid", "class"]).set_index('annoid')
       
#         cells = pd.DataFrame(cur.execute(f"""SELECT coordinateX, coordinateY, annoId
#                                 from Annotations_coordinates where slide=={slide[0]}""").fetchall(),\
#                                     columns=['x', 'y', 'annoid'])

#         idx = -1
        
#         for col in range(0,slide[2]-IMGSZ,IMGSZ):
#             for row in range(0,slide[1]-IMGSZ,IMGSZ):
#                 idx += 1

#                 if idx % 500 == 0:
#                     print(file_count, "/", file_len, file_name,":",idx, "/", tiles)

#                 location = (row, col)
#                 local_cells = cells[['x','y','annoid']][cells['x'] > location[0]][cells['y'] > location[1]]\
#                     [cells['x']<location[0]+IMGSZ][cells['y']<location[1]+IMGSZ]
#                 local_cells['x'] -= location[0]
#                 local_cells['y'] -= location[1]

#                 flag = 0
#                 for cell in local_cells.values.tolist():
#                     try:
#                         label = Annotations.loc[cell[2]].values[0]
#                     except:
#                         print(cell)
#                         continue
#                     # 원래 x1, y1, x2, y2이지만 편의상 w, h로 표기
#                     x, y, w, h = cell[0]-cell_size//2 , cell[1]-cell_size//2, cell[0] + cell_size//2, cell[1] + cell_size//2

#                     if label == 7:
#                         label = 0

#                     #imgsize 범위 벗어나는 경우, label 0 ~ 3이 아닌 경우 pass
#                     if label >= 4 or x <= 0 or y <= 0 or w >= IMGSZ or h >= IMGSZ:
#                         continue
                    
#                     img_name, label = "images/"+file_name+"_"+str(idx)+'.dcm ',  LABELS[label]

#                     line = dest_dir+"/"+img_name + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "," + label + "\n"
#                     f.write(line)
#                     flag = 1

#                 if flag == 0:
#                     f.write(dest_dir+"/images/"+file_name+"_"+str(idx)+'.dcm ' + ",,,,,\n")

#                 # # 이미지 저장
#                 # img = Image.fromarray(ds.read_region(location=location,size=(IMGSZ,IMGSZ)))
#                 # img.save(dest_dir + "/images/" + file_name + "_" + str(idx) + ".png", 'png')
#     f.close()


## too slow  
# def yolo_dcm_to_train_set(db = "original_data/archive/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite", source_dir = "original_data/archive",\
#                          dest_dir = "datasets/new", tile_size = 640, cell_size = 40):
    
#     if not os.path.isdir(dest_dir):                                                           
#         os.mkdir(dest_dir)
#         os.mkdir(dest_dir+"/images")
#         os.mkdir(dest_dir+"/labels")

#     DB = sqlite3.connect(db)
#     cur = DB.cursor()
#     IMGSZ = tile_size
#     datasets = glob(source_dir +"/*.dcm")

#     file_names = []
#     for data in datasets:
#         temp = data.split("\\")[1]
#         file_names.append(temp.split('.')[0])

#     file_names.sort()
#     files = len(file_names)
#     file_count = 0

#     for file_name in  file_names:
#         file_count += 1
        
#         # filename -> slide 번호, width, height 추출
#         slide = cur.execute(f"""SELECT uid, width, height
#                                 from Slides 
#                                 where filename == "{file_name+".dcm"}" """).fetchall()
#         slide = slide[0]

#         save_dir = dest_dir + '/'
        
#         # 이미지 읽기 위한 Dicom Dataset 객체 객체 생성
#         ds = ReadableDicomDataset(source_dir+ "/" + file_name + ".dcm")

#         tiles = ds.geometry_imsize
#         tiles = int(tiles[0]/IMGSZ) * int(tiles[1]/IMGSZ)

#         idx = -1
#         for height in range(0,slide[2]+IMGSZ,IMGSZ):
#             for width in range(0,slide[1]+IMGSZ,IMGSZ):
#                 idx += 1
#                 txt_file = save_dir + "labels/" + file_name + "_" + str(idx) + ".txt"

#                 if idx % 500 == 0:
#                     print(file_count, "/", files, file_name,":",idx, "/", tiles)
#                 location = (width, height)
#                 size = (IMGSZ,IMGSZ)
#                 # cells = (현재 location에서의 x, y, uid)
#                 cells = cur.execute(f"""SELECT coordinateX-{location[0]}, coordinateY-{location[1]}, annoId
#                                 from Annotations_coordinates where slide=={slide[0]} and 
#                                 coordinateX>{location[0]} and coordinateX<{location[0]+size[0]} and 
#                                 coordinateY>{location[1]} and coordinateY<{location[1]+size[1]}""").fetchall()
#                 if len(cells) > 0:
#                     lines = []
#                     for cell in cells:
#                         # Annotations_coordinates에는 존재하지만 해당 annoid가 Annotations에 존재하지 않는 경우 저장하지 않음.
#                         try:
#                             cell_class = cur.execute(f"""SELECT agreedClass FROM Annotations where uid == {cell[2]}""").fetchall()[0][0]

#                             label, x, y, w, h = cell_class, cell[0]/IMGSZ, cell[1]/IMGSZ, cell_size/IMGSZ, cell_size/IMGSZ

#                             if label == 7:
#                                 label = 0

#                             if x - w/2 <= 0 or x + w/2 >= 1 or  y - h/2 <= 0 or y + h/2 >= 1 or label == 4:
#                                 continue

#                             line = str(label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
#                             lines.append(line)
#                         except:
#                             None

#                     if len(lines) > 0:
#                         with open(txt_file, 'w')as f:
#                             for line in lines:
#                                 f.write(line)

#                 if os.path.exists(txt_file):
#                     img = Image.fromarray(ds.read_region(location=location,size=size))
#                     img.save(save_dir + "images/" + file_name + "_" + str(idx) + ".png", 'png')


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
                if temp[0] == 7 or temp[0] == '7':
                    print(temp)
                cells.append(temp)

    cells = pd.DataFrame(cells, columns=["label", "x", "y", "w", "h"])
    return cells
