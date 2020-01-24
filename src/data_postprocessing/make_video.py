import argparse
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import HTML
from skimage import io

matplotlib.rcParams['animation.embed_limit'] = 2**128

dir_name = os.path.dirname(__file__)


def make_video(source_dir, results_dir, filename="output"):

    source_img_dir = Path(os.path.join(source_dir, 'test_img'))
    target_dir = Path(os.path.join(results_dir, 'test_latest/images'))
    label_dir = Path(os.path.join(source_dir, 'test_label_ori'))

    source_img_paths = sorted(source_img_dir.iterdir())
    target_synth_paths = sorted(target_dir.glob('*synthesized*'))
    target_label_paths = sorted(label_dir.iterdir())

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    def render_figure(nframe):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        source_img = io.imread(source_img_paths[nframe])
        ax1.imshow(source_img)
        ax1.set_xticks([])
        ax1.set_yticks([])

        target_label = io.imread(target_label_paths[nframe])
        ax2.imshow(target_label)
        ax2.set_xticks([])
        ax2.set_yticks([])

        target_synth = io.imread(target_synth_paths[nframe])
        ax3.imshow(target_synth)
        ax3.set_xticks([])
        ax3.set_yticks([])

    video_name = os.path.join(results_dir, "{}.mp4".format(filename))

    for idx, _ in tqdm(enumerate(target_synth_paths)):
        render_figure(idx)
        fig.canvas.draw()

        if idx == 0:
            width, height = fig.canvas.get_width_height()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*'VP90')

            video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare target Video')
    parser.add_argument('-s', '--source-dir', type=str,
                        default=os.path.join(dir_name, '../../data/sources/bruno_mars'),
                        help='Path to the folder where the source video is saved. One video per folder!')
    parser.add_argument('-r', '--results-dir', type=str,
                        default=os.path.join(dir_name, '../../results/example_bruno_mars'),
                        help='Path to the folder where the results of transfer.py are saved')
    args = parser.parse_args()
    make_video(args.source_dir, args.results_dir)
