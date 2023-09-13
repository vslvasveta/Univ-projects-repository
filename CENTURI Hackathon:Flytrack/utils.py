from importlib.metadata import PackageNotFoundError
import numpy as np
import shutil
import os
import glob
from tqdm import tqdm

import requests
import zipfile
import shutil

from PIL import Image, ImageDraw, ImageFont
import tifffile
import skimage.draw as sk_draw


import time
from datetime import datetime as datetime

import napari
import cv2


def folder_tifs_to_single_tif(path_to_dir: str, dtype='uint16'):

    files = listdir(path_to_dir, return_fullpath=True)

    final_tif = []

    for file in files:
        if file.endswith('.tif'):
            final_tif.append(tifffile.imread(file))

    return np.array(final_tif, dtype=dtype)

 
def import_data_from_AMUBox(folder_path: str, movie_folder: str):

    # url of the data
    url=requests.get('https://amubox.univ-amu.fr/s/wCeHGMRNcz29nbo/download')

    # extension of the files
    ext = ".zip"

    # Create the zip file
    filename = "hackathon_data"

    with open(f'{folder_path}/' + filename + ext, mode="wb") as localfile:
        localfile.write(url.content)

    # Extract images from the ZIP file
    with zipfile.ZipFile(f'{folder_path}/' + filename + ext, 'r') as zip_ref:
        zip_ref.extractall(f'{folder_path}' + '/.')

    # extract files of the movie
    with zipfile.ZipFile(f'{folder_path}/' + filename + '/' + movie_folder + ext, 'r') as zip_ref:
        zip_ref.extractall(f'{folder_path}/')


def detections_to_segmentation(detections, segmentation_shape: tuple, sizes):

    offset = 10
    segmentation_shape_offset = tuple(elem + 2* offset for elem in segmentation_shape)

    segmentation_offset = np.zeros(shape=segmentation_shape_offset, dtype='uint16')

    for index, (detection, size) in enumerate(zip(detections, sizes), start=1):
        
        detection_offset = (elem+offset for elem in detection)

        rr, cc = sk_draw.disk(detection_offset, size)
        segmentation_offset[rr, cc] = index

    segmentation = segmentation_offset[offset:-offset, offset:-offset]

    return segmentation

def dtime(with_day: bool = False):

    dtime = datetime.fromtimestamp(time.time())

    if not with_day:
        dtime = str(dtime).split(' ')[1]

    return str(dtime)


def embed_time_on_array(array):

    img = Image.fromarray(array)
    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)
    # Custom font style and font size
    myFont = ImageFont.truetype('arial.ttf', 20)
    # Add Text to an image
    I1.text((10, 10), f'{dtime()}', fill=(255,255,255), font=myFont)

    return np.asarray(img)


def listdir(path_to_dir: str, return_fullpath: bool = False):

    files = os.listdir(path_to_dir)

    if return_fullpath:

        abs_path = os.path.abspath(path_to_dir)
        files = list(map(lambda x: os.path.join(abs_path, x), files))

    return sorted(files)

def rgb2gray(rgb_array, output_integers: bool = True):

    gray_array = np.dot(rgb_array[...,:3], [0.2989, 0.5870, 0.1140])

    if output_integers:
        # Convert array values to integers and clip them
        # between 0 and 255 
        gray_array = convert_to_int_and_clip_array(gray_array)
    
    return gray_array

def convert_to_int_and_clip_array(array):
    """
    Convert array values to integers and clip them
    between 0 and 255.
    """ 

    array = np.clip(
                        np.round(array), 
                        a_min=0, 
                        a_max=255
                    ).astype('uint8')

    return array

def time_to_filename(time_as_float: float, prefix: str = 'img_', 
                     suffix: str = '.tif') -> str:

    str_time = str(time_as_float).replace('.', '_')
    filename = f'{prefix}{str_time}{suffix}'

    return filename

def filename_to_time(filename: str, prefix: str = 'img_', 
                     suffix: str = '.tif') -> float:

    str_time = filename[len(prefix): -len(suffix)]
    time_as_float = float(str_time.replace('_', '.'))

    return time_as_float

def delete_content_of_dir(path_to_dir: str, content_type: str = ''):

    files = glob.glob(f'{path_to_dir}/*{content_type}')
    
    print(f'Deleting content of folder {path_to_dir}')
    for f in tqdm(files):
        try:
            os.remove(f)
        except IsADirectoryError:
            shutil.rmtree(f)



def detections_napari(all_detections):

    '''
    Convert detection into format for napari
    
    Params:
        all_detections: List of detections [num_steps, detections] where detections have shape (num_detections, 2)
        
    Returns:
        Numpy array of shape (num_total_detections, 3) with t, x, y for every detection
    '''

    detections_napari = []

    for frame_index, detections in enumerate(all_detections): # loop on frames
        for elem in detections:
            detections_napari.append([frame_index, *elem])

    return detections_napari



def tracks_napari(tracks):

    tracks_napari = []

    for ind, track in enumerate(tracks):
        for track_bits in track:
            tracks_napari.append([ind, *track_bits])

    return np.array(tracks_napari)


def mp4toimgs(path_movie, out_folder):

    capture = cv2.VideoCapture(path_movie)
    curr_frame = 0

    while True:

        _, frame = capture.read()

        if frame is None:
            break

        name = os.path.join(out_folder, 'frame') + str(curr_frame).zfill(4) + '.tif'
        cv2.imwrite(name, frame)

        curr_frame += 1

    capture.release()
    cv2.destroyAllWindows()



def remove_background(path_movie, out_folder):

    back_sub = cv2.createBackgroundSubtractorKNN(history=3000, dist2Threshold=4000)
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(path_movie))

    curr_frame = 0

    while True:

        ret, frame = capture.read()

        if ret:
            mask = back_sub.apply(frame)

            name = os.path.join(out_folder, 'frame') + str(curr_frame).zfill(4) + '.tif'  
            cv2.imwrite(name, mask)

        else:
            break
        
        curr_frame += 1