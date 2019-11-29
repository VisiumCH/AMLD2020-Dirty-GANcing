import argparse
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from skimage import io

matplotlib.rcParams['animation.embed_limit'] = 2**128

dir_name = os.path.dirname(__file__)


def make_gif(source_dir, results_dir):

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

    def animate(nframe):
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

    anim = ani.FuncAnimation(fig, animate, frames=len(target_label_paths), interval=1000 / 24)
    plt.close()

    #js_anim = HTML(anim.to_jshtml())
    
    anim.save(os.path.join(results_dir, "test_latest/output.gif"), writer="imagemagick")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare target Video')
    parser.add_argument('-s', '--source-dir', type=str, default=os.path.join(dir_name, '../../data/sources/bruno_mars'),
                        help='Path to the folder where the source video is saved. One video per folder!')
    parser.add_argument('-r', '--results-dir', type=str, default=os.path.join(dir_name, '../../results/example_bruno_mars'),
                        help='Path to the folder where the results of transfer.py are saved')
    args = parser.parse_args()
    make_gif(args.source_dir, args.results_dir)
