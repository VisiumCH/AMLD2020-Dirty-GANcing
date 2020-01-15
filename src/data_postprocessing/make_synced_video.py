import argparse
import os
import math
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


def make_synced_video(source_dir, results_dirs):

    target_dir_list = [Path(os.path.join(results_dir, 'test_latest/images')) for results_dir in results_dirs]

    target_synth_paths = [sorted(target_dir.glob('*synthesized*')) for target_dir in target_dir_list]

    target_num = len(target_dir_list)
    grid_size = math.ceil(math.sqrt(target_num))
    result_figsize_resolution = 40 # 1 = 100px

    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(result_figsize_resolution, result_figsize_resolution), 
                             sharex=True, sharey=True)

    def render_figure(nframe):
        for ax in axes.flatten():
            ax.clear()
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

        for idx, target in enumerate(target_synth_paths):
            x_position = idx % grid_size
            y_position = idx // grid_size

            plt_image = io.imread(target_synth_paths[idx][nframe])
            axes[x_position, y_position].imshow(plt_image)
        fig.subplots_adjust(hspace=0, wspace=0)

    video_name = os.path.join('results', 'synced', "synced_output.mp4")

    for idx, _ in tqdm(enumerate(target_synth_paths[0])):
        render_figure(idx)
        fig.canvas.draw()

        if idx == 0:
            width, height = fig.canvas.get_width_height()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    parser.add_argument('-r', '--results-dirs', type=str, nargs='+',
                        default=os.path.join(dir_name, '../../results/example_bruno_mars'),
                        help='Path to the folder where the results of transfer.py are saved')
    args = parser.parse_args()
    make_synced_video(args.source_dir, args.results_dirs)
