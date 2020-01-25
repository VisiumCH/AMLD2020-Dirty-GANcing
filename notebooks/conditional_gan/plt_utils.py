import time
import matplotlib.pyplot as plt


def init_figure():
    """Init interactive figure with 3 subplots for the notebook graphics"""
    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    #plt.ion()

    #fig.show()
    return fig, (ax1, ax2, ax3)


def plot_current_results(image, model_loss, fig, axes):
    """Use the visuals dict to fill the fig and axes"""
    ax1, ax2, ax3 = axes

    ax1.clear()
    ax2.clear()
    ax3.clear()

    epochs = range(1, len(model_loss['generator'])+1)
    
    ax1.plot(epochs, model_loss['generator'])
    ax1.set_title('Generator Loss')

    ax2.plot(epochs, model_loss['discriminator'])
    ax2.set_title('Discriminator Loss')

    ax3.imshow(image)
    ax3.set_xticks([])
    ax3.set_yticks([])
    #fig.canvas.draw()
    plt.show()
