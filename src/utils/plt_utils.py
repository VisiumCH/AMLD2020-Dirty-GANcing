import time
import matplotlib.pyplot as plt


def init_figure():
    """Init interactive figure with 3 subplots for the notebook graphics"""
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #plt.ion()

    #fig.show()
    return fig, (ax1, ax2, ax3)


def plot_current_results(visuals, fig, axes):
    """Use the visuals dict to fill the fig and axes"""
    ax1, ax2, ax3 = axes

    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.imshow(visuals['real_image'])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(visuals['input_label'])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(visuals['synthesized_image'])
    ax3.set_xticks([])
    ax3.set_yticks([])
    #fig.canvas.draw()
    plt.show()
