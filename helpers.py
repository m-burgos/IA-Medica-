import os
import xml.etree.ElementTree as et
from PIL import Image, ImageDraw
import cv2
import numpy as np
from openslide import OpenSlide

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#retorna lista con todos los .xml/.svs en <path>
def filenames(path, extension):
    if extension == 'svs':
        return sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) if f[-3:]=='svs'])
    elif extension == 'xml':
        return sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) if f[-3:]=='xml'])
    else:
        print('extension: svs ó xml !')        
        
def read_svs(path, fn):
    return OpenSlide(os.path.join(path, fn[:-3] + 'svs'))  
        
#input: pol = [(x1,y1)...(xn,yn)] & size=slide.dimensions  ||  output: Min_x, Max_x, Min_y, Max_y
def boundbox_pol(pol, size): 
    Max_x, Min_x = max(point[0] for point in pol), min(point[0] for point in pol)
    Max_y, Min_y = max(point[1] for point in pol), min(point[1] for point in pol)
    return max(0, Min_x), min(Max_x, size[0]), max(0, Min_y), min(Max_y, size[1])
     
#input: size (slide.dimensions) & pols (lista donde c/elemento es la lista de ptos de 1 poligono) -> output: mask (blanco & negro)
def whole_mask(pols, size):
    img = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.fillPoly(img, pols, color=255) 
    return img

#input: img & (left, top) coordinates & dim (pixel) del parche -> output: img cropped
def crop_mask(img, point, pix):
    #return  img.crop( (point[0], point[1], point[0]+pix, point[1]+pix) )
    return  img[point[1]:point[1]+pix, point[0]:point[0]+pix]

#input coordenadas en pixel -> retorna coordenadas en micro metro | c = [micro m / pixel]
def wsi_coord(x,y,c):
    #return int(round(x*c,0)), int(round(y*c,0))
    return str(int(round(x*c,0))), str(int(round(y*c,0))) #puesto que lo usamos solo para poner en el filename del parche

def count_zero_rows_columns(array): #numpy_array!
    zero_row_count = np.sum(np.all(array == 0, axis=1))
    zero_column_count = np.sum(np.all(array == 0, axis=0))
    return max(zero_row_count, zero_column_count)

#percent: desde que porcentaje se considera que un parche está en blanco
#threshold:  desde que valor (grayscale) es blanco un pixel
def is_patch_blank(patch, percent=0.6, threshold=200): 
    np_patch = np.array(patch.convert('L'))
    w_per_patch = np.sum((np_patch > threshold))/len(np_patch)**2 
    return ((np.std(np_patch) < 25)&(np.mean(np_patch)>threshold)) or (count_zero_rows_columns(np_patch) > 10) or (w_per_patch > percent)


#desplazar las coordenadas de los poligonos        
def update_coord(poligonos, x=0, y=0):
    return [np.array([(punto[0]-x, punto[1]-y) for punto in poligono], dtype=np.int32) for poligono in poligonos]
        
#retorna una lista donde cada elemento son las poligonales registradas en fn (filename) en el color dado
def get_vert(fn, color):
    root, v = et.parse(fn).getroot(), []
    for a in root:
        if a.get('LineColor') == str(color):
            for region in a[1][1:]:
                #v.append( [( int(float(c.get('X'))),  int(float(c.get('Y'))) )  for c in region[1]] )
                v.append( np.array([( int(float(c.get('X'))),  int(float(c.get('Y'))) )  for c in region[1]]) )
    return v

# devuelve diccionario con Todas las poligonales de <color> para cada .xml en <path>
def get_all_vert(path, color):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) if f[-3:]=='xml']
    return {f[:-4]: get_vert(os.path.join(path,f), color) for f in files}

def get_xy(data): #input: [(x1,y2)...(xn,yn)] ==> output: [x1...xn], [y1...yn]
    return [p[0] for p in data], [p[1] for p in data]
