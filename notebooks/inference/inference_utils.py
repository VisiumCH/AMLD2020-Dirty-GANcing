import argparse
import os
import datetime
import json
import numpy as np
import sys
import cv2
import time

from tqdm import tqdm
from ipywidgets import Video

dir_name = os.path.abspath('.')

sys.path.append(os.path.join(dir_name, '../../src'))
sys.path.append(os.path.join(dir_name, '../../src/PoseEstimation/'))
sys.path.append(os.path.join(dir_name, '../../src/utils'))

from data_preparation.prepare_source import prepare_source
from data_postprocessing.normalization import normalize
from data_postprocessing.transfer import test_transfer
from data_postprocessing.make_video import make_video

def GAN_inference(target_name, run_name, target_runname, already_recorded):
    record_and_makevideo(target_name, run_name, target_runname, already_recorded)
    
    video = Video.from_file(os.path.join(dir_name, '../../results/',run_name, "output.mp4"), play=True)
    return video

def record_and_makevideo(target_name, run_name, target_runname, already_recorded):
    source_dir = os.path.join(dir_name, '../../data/sources', run_name)
    os.makedirs(source_dir, exist_ok=True)

    target_dir = os.path.join(dir_name, '../../data/targets/', target_name)

    if not already_recorded:
        countdown(t=5)

        record_video(source_dir)

        prepare_source(source_dir, frames=True)

    normalize(source_dir, target_dir)

    os.makedirs(os.path.join(dir_name, '../../checkpoints/', run_name), exist_ok=True)

    test_transfer(source_dir, target_runname, live_run_name=run_name)

    make_video(source_dir, os.path.join(dir_name, '../../results', run_name))


def record_video(source_dir):
    img_dir = os.path.join(source_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    stream = cv2.VideoCapture('http://127.0.0.1:8080')

    # Check if the webcam is opened correctly
    if not stream.isOpened():
        raise IOError("Cannot open webcam")

    for i in tqdm(range(100), 'acquiring frames'):
        ret, frame = stream.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(img_dir, '{:05d}.png'.format(i)), frame)

    stream.release()


def countdown(t):
    while t > 0:
        print(t)
        t -= 1
        if t == 0:
            print('DANCING TIME!')
            break
        time.sleep(1)
