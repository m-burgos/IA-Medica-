import os
import cv2
import argparse
import pickle 
import pandas as pd
from PIL import Image
from helpers import *
from time import time
from openslide import PROPERTY_NAME_MPP_X
T = time()

#################################################################
# run using the command
# python m_all.py -pix <pix>
# where <pix> can be 224, 320, or any integer pixel size for the patches.
#################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-pix', '--pixel', type=int, help='Pixel size of patches ')
args = parser.parse_args()

COLOR = {'R': 255, 'V': 65408, 'C': 16776960}
pix = args.pixel
#################################################################
#
# Output directory has the structure:
# M_dataset<pix>
#   |_patch
#   |_mask
#   |   |_red       
#   |   |_green
#   |_temp 	    ---> saves masked patches for WSI reconstruction
#   |_coord.parquet ---> (file, x, y) referent to the bounding box
#   |_data.pkl      ---> {file: {point:, size:}} vertex and size bounding box
#
#################################################################
#Set path for the svs files here
in_path = '/home/user/Documentos/HHHA-2018-01895_Dr.JCA' 
out_path = 'M_dataset'+str(pix)                                   
#################################################################
path_mask_r = os.path.join(out_path, 'mask', 'red')
path_mask_v = os.path.join(out_path, 'mask', 'green')
path_patch = os.path.join(out_path, 'patch')
alfa = 0.35 # <--- esto también podría ser ingresado via argparse

path_coords_df = os.path.join(out_path, 'coords.parquet')
path_data_dict = os.path.join(out_path, 'data.pkl')
path_masked_patches = os.path.join(out_path, 'temp')
#################################################################

def crop(pix, in_path, out_path, formato='.png'):
	check_dir(path_mask_r)
	check_dir(path_mask_v)            #crea las carpetas donde guardaremos
	check_dir(path_patch)             #los patch&mask (en caso que no existan)
	files = filenames(in_path, 'xml')
	patches_per_file, d = [], {}
	X, Y, i = [], [], 0
	t = time()
	for file in files:
	    slide = read_svs(in_path, file)
	    pols_r = get_vert(os.path.join(in_path, file), COLOR['R'])
	    pols_v = get_vert(os.path.join(in_path, file), COLOR['V'])
	    pols_c = get_vert(os.path.join(in_path, file), COLOR['C'])[0]
	    xmin,xmax,ymin,ymax = boundbox_pol(pols_c, slide.dimensions) #Aquí se determina el nuevo origen
	    
	    width, height = xmax-xmin, ymax-ymin
	    d[file[:-4]] = {'point':(xmin,ymin), 'size':(width, height)}
	    pols_r, pols_v = update_coord(pols_r, x=xmin, y=ymin), update_coord(pols_v, x=xmin, y=ymin)
	    wsi_mask_r, wsi_mask_v = whole_mask(pols_r, (width, height)), whole_mask(pols_v, (width, height))  
	    
	    c = float(slide.properties.get(PROPERTY_NAME_MPP_X))
	    t0 = time()
	    patches_count = 0

	    for y in range(ymin, ymax, pix):
	        for x in range(xmin, xmax, pix):
	            patch = slide.read_region((x, y), 0, (pix, pix)).convert('RGB')
	            if not is_patch_blank(patch):
	                mask_r = crop_mask(wsi_mask_r, (x-xmin,y-ymin), pix)
	                mask_v = crop_mask(wsi_mask_v, (x-xmin,y-ymin), pix)
	                xx, yy = wsi_coord(x,y,c)                                 #coordenadas [micrometro]
	                filename = f"{i:05d}_{xx}_{yy}"+formato                   #no trasladamos, solo para ver en software (referencia)
	                cv2.imwrite(os.path.join(path_mask_r, filename), mask_r)
	                cv2.imwrite(os.path.join(path_mask_v, filename), mask_v)
	                patch.save(os.path.join(path_patch,filename))
	                patches_count += 1
	                X.append(x-xmin), Y.append(y-ymin)                        #coord relativas al nuevo origen! [pixel]
	                i += 1

	    patches_per_file.append(patches_count)
	    print('{} cropped in {:.2f} min '.format(file[:-4], (time()-t0)/60) )

	print(f'Total patches: {len(X)}')
	print(f'{patches_per_file = } ')
	files = [xml[:-4] for xml in files]
	file_list = [fn for file, n in zip(files, patches_per_file) for fn in [file]*n]

	import pandas as pd
	df = pd.DataFrame({'file': file_list, 'x': X, 'y': Y}) #coordenadas [pixel]
	df.to_parquet(os.path.join(out_path, 'coords.parquet'))
	with open(os.path.join(out_path, 'data.pkl'), 'wb') as f:
		pickle.dump(d, f)
	print(f'All files cropped in {(time()-t)/60:.2f} min ')


def patch_masking(patch, mask, color, alfa):
    color = np.array(color, dtype=np.uint8)
    imagen_con_color = np.zeros(patch.shape, dtype=np.uint8)
    imagen_con_color[:] = color
    imagen_con_color[mask != 255] = patch[mask != 255] 
    imagen_final = cv2.addWeighted(patch, 1 - alfa, imagen_con_color, alfa, 0)
    return imagen_final

def redgreen_patch(patch, mask_red, mask_green, alfa):
    img1 = patch_masking(patch, mask_red, (0,0,255), alfa)
    return patch_masking(img1, mask_green,(0,255,0), alfa)

def masking_all_patches(path_mask_r, path_mask_v, path_patch, path_masked_patches, alfa):
    t0 = time()
    check_dir(path_masked_patches)
    files = sorted(os.listdir(path_mask_r))
    for file in files:
        mask_r = cv2.imread(os.path.join(path_mask_r, file))
        mask_v = cv2.imread(os.path.join(path_mask_v, file))
        patch = cv2.imread(os.path.join(path_patch, file))
        if np.sum(mask_r) != 0 :
        	if np.sum(mask_v) != 0:
        		img = redgreen_patch(patch, mask_r, mask_v, alfa)
        	else:
        		img = patch_masking(patch, mask_r, (0,0,255), alfa)
        elif np.sum(mask_v) != 0:
        	img = patch_masking(patch, mask_v, (0,255,0), alfa)
        else:
            img = patch
        cv2.imwrite(os.path.join(path_masked_patches, file), img)
    print(f'Masked all patches in {(time()-t0):.2f} sec')


def reconstruct_wsi(filename, path_masked_patches, path_coords_df, full_image_size):
    full_image = Image.new('RGB', full_image_size, (240,240,240))
    df = pd.read_parquet(path_coords_df)
    df = df[df.file == filename]
    files = sorted(os.listdir(path_masked_patches))
    for idx, row in df.iterrows():
        patch = Image.open(os.path.join(path_masked_patches, files[idx])) #leemos la mask 
        full_image.paste(patch, (row.x, row.y))         #pegamos sobre fondo 
    
    return full_image

#################################################################
print(30*'-'+str(pix)+30*'-')

crop(pix, in_path, out_path)
masking_all_patches(path_mask_r, path_mask_v, path_patch, path_masked_patches, alfa)

df = pd.read_parquet(path_coords_df)
with open(path_data_dict, 'rb') as f:
    d = pickle.load(f)

for fn in df.file.unique():
	t0 = time()
	img = reconstruct_wsi(fn, path_masked_patches, path_coords_df, d[fn]['size'])
	img.save(str(pix)+'_'+fn+'_masked.png')
	print(f'{fn} reconstructed & saved in {(time()-t0):.2f} sec')
	del img

print(f'All has been done in {(time()-T)/60:.2f} min\n')
