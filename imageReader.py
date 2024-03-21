import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime
import time

start_time = time.time()

OPENSLIDE_PATH = r'D:\Usuario\Desktop\ProjetoMestrado\openslide-win64-20231011\openslide-win64-20231011\bin'
path_Image=r"D:\Usuario\Desktop\ProjetoMestrado\Image_test\HHHA-2018-01895 B2 00-40.svs"
path_Annotation=path_Image.replace('.svs','.xml',1)
path_cancer_folder=r"D:\Usuario\Desktop\ProjetoMestrado\Image_test\CANCER"
path_not_cancer_folder=r"D:\Usuario\Desktop\ProjetoMestrado\Image_test\NOT_CANCER"
COUNTER_PATIENT=3

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

svs_image = openslide.OpenSlide(path_Image)


print("Dimensions:", svs_image.dimensions)
print("Number of levels:", svs_image.level_count)
print("Level dimensions (width, height) for level 0:", svs_image.level_dimensions[0])


# Read annotations from XML file
annotations_tree = ET.parse(path_Annotation)
annotations_root = annotations_tree.getroot()

#defining Annotation list
annotations_cancer = []
annotations_not_cancer=[]

for annotation in annotations_root.findall('Annotation'):
    # Extract coordinates and other relevant information
    # This will depend on the structure of your XML file
    if annotation.get('LineColor') == "255":
        layer_region=annotation.findall("Regions//Region")
        for region in layer_region:
            temp=[]
            vertices=region.findall("Vertices//Vertex")
            for vertex in vertices:
                temp_coordinates=[]  
                x=float(vertex.get("X"))
                y=float(vertex.get("Y"))
                temp_coordinates.append(x)
                temp_coordinates.append(y)
                temp.append(tuple(temp_coordinates))
            annotations_cancer.append(temp)
    elif annotation.get('LineColor') == "65408":
        layer_region=annotation.findall("Regions//Region")
        for region in layer_region:
            temp=[]
            vertices=region.findall("Vertices//Vertex")
            for vertex in vertices:
                temp_coordinates=[] 
                x=float(vertex.get("X"))
                y=float(vertex.get("Y"))
                temp_coordinates.append(x)
                temp_coordinates.append(y)
                temp.append(tuple(temp_coordinates))
            annotations_not_cancer.append(temp)
    else:
        pass

window_size = 320
stride = 160
counter_cancer=0
counter_not_cancer=0

image_width, image_height = svs_image.dimensions
total_windows = ((image_width - window_size) // stride + 1) * ((image_height - window_size) // stride + 1)


for y in range(0, svs_image.dimensions[1], stride):
    for x in range(0, svs_image.dimensions[0], stride):
        # Define the window
        window = svs_image.read_region((x, y), 0, (window_size, window_size))
        window_np = np.array(window)
        # Check for intersections
        for annotation in annotations_cancer:
            # Convert annotation coordinates to a Shapely Polygon
            # This will depend on how your annotations are stored
            polygon = Polygon(annotation)
            # Calculate intersection
            intersection = polygon.intersection(Polygon([(x, y), (x+window_size, y), (x+window_size, y+window_size), (x, y+window_size)]))
            # Check if the intersection area is larger than 75% of the window area
            if intersection.area / ((window_np.size)/svs_image.level_count) > 0.75:
                counter_cancer=counter_cancer+1
                # Extract and save the patch
                # This will depend on how you want to save the patches
                # Convert the NumPy array to a PIL Image
                patch_image = Image.fromarray(window_np)
                # Save the image as a PNG file
                # You might want to generate a unique filename for each patch
                now = datetime.now()
                timestamp = now.strftime('%m_%d_%Y_%H_%M_%S')
                file_path = "CANCER_PATIENT_"+str(COUNTER_PATIENT)+"_IMAGE_"+str(counter_cancer)+'_'+str(timestamp)+'.png'
                patch_image.save(path_cancer_folder+'/'+file_path)
        for annotation in annotations_not_cancer:
            # Convert annotation coordinates to a Shapely Polygon
            # This will depend on how your annotations are stored
            polygon = Polygon(annotation)
            # Calculate intersection
            intersection = polygon.intersection(Polygon([(x, y), (x+window_size, y), (x+window_size, y+window_size), (x, y+window_size)]))
            # Check if the intersection area is larger than 75% of the window area
            if intersection.area / ((window_np.size)/svs_image.level_count) > 0.75:
                counter_not_cancer=counter_not_cancer+1
                # Extract and save the patch
                # This will depend on how you want to save the patches
                # Convert the NumPy array to a PIL Image
                patch_image = Image.fromarray(window_np)
                # Save the image as a PNG file
                # You might want to generate a unique filename for each patch
                now = datetime.now()
                timestamp = now.strftime('%m_%d_%Y_%H_%M_%S')
                file_path = "NOT_CANCER_PATIENT_"+str(COUNTER_PATIENT)+"_IMAGE_"+str(counter_not_cancer)+'_'+str(timestamp)+'.png'
                patch_image.save(path_not_cancer_folder+'/'+file_path)       
    processed = (y // stride) * ((image_width - window_size) // stride + 1) + (x // stride) + 1
    print(f"Progress: {processed}/{total_windows} windows processed ({processed / total_windows:.2%})", end='\r')
   

# Close the SVS image
svs_image.close()

end_time = time.time()
execution_time = end_time - start_time
execution_time=execution_time/60
print(f"Program executed in {execution_time} minutes")

