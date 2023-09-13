import os
import glob
import napari
import numpy as np
from PIL import Image

from detection.log_detector import *
from tracking.euclidian_tracker import *
from utils import *

import argparse


def detect(path2imgs):

    '''
    Args:
        path2imgs: Folder containing the sorted images as .tif files

    Returns:
        List of detections [num_steps, num_detections_at_t, 2]
    '''

    os.path.join(path2imgs, '*.tif')
    img_files = sorted(glob.glob(os.path.join(path2imgs, '*.tif')))

    all_detections = []

    for file in img_files:

        gray_img = np.array(Image.open(file))

        detector = LoGDetector()
        detections = detector.detect(
            gray_img,
            sigma=(1,5),
            num_sigma=3,
            threshold=20    # 20
        )

        all_detections.append(detections)

    return all_detections




def track(all_detections, observation_noise, transition_noise,
          trackability: bool = True, n_sampling: int = 20, dt = 1):

    tracker = EuclidianTracker(trackability=trackability, n_sampling=n_sampling)

    tracks, trackabilities = tracker.track(
        all_detections,
        observation_noise = observation_noise,      # THOSE PARAMETERS ARE RUBBISH, DEFINE THEM PROPERLY
        transition_noise = transition_noise,    
        states_initial_covariance = np.eye(4)*10,                           # THOSE PARAMETERS ARE RUBBISH, DEFINE THEM PROPERLY
        list_dts = [dt] * (len(all_detections)-1)
    )

    return tracks, trackabilities, tracker.appearances, tracker.disappearances





def detect_and_track(path2imgs, observation_noise=5, transition_noise=10,
                     display=True, n_sampling=20):

    all_detections = detect(path2imgs)
    tracks, trackability, _, _ = track(all_detections, observation_noise, transition_noise, n_sampling=n_sampling)

    print('trackability: ', trackability)

    trs_napari = None
    det_napari = None
    
    if display:
        det_napari = detections_napari(all_detections)
        trs_napari = tracks_napari(tracks)
        movie = folder_tifs_to_single_tif(path2imgs)
        viewer = napari.Viewer()
        viewer.add_image(movie)
        viewer.add_points(np.array(det_napari), size=5, face_color='red', name='detections')
        viewer.add_tracks(np.array(trs_napari), name='tracks')
        napari.run()

    return trs_napari, det_napari, np.mean(trackability)



def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d','--dir', help='directory')
    parser.add_argument('-obs_noise','--observation_noise', default=5, type=float, help='observation noise')
    parser.add_argument('-tr_noise','--transition_noise', default=10, type=float, help='transition noise')
    parser.add_argument('-n','--n_sampling', type=int, default=5, help='Number of Monte Carlo samplings for trackability estimation')
    args = vars(parser.parse_args())

    detect_and_track(path2imgs=args['dir'],
                     observation_noise=args['observation_noise'],
                     transition_noise=args['transition_noise'],
                     display=True,
                     n_sampling=args['n_sampling'])



if __name__ == "__main__":
    main()