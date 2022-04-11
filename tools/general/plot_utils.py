# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import numpy as np
import seaborn as sns

from matplotlib import cm
from matplotlib import pyplot as plt

def draw_plot(
    x_data, y_data, 
    label='Default', color='b', 

    title='', 
    image_path='', 

    clear=False,
    show=True,
    ):
    if clear:
        plt.clf()
    
    plt.plot(x_data, y_data, color=color, label=label)
    if label is not 'Default':
        plt.legend()

    if title != '':
        plt.title(title)

    if image_path != '':
        plt.savefig(fname=image_path)
        
    if show:
        plt.show()

def draw_3d_plot(data):
    # sns.set_theme(style="darkgrid")

    h, w = data.shape
    x1 = np.arange(w)
    x2 = np.arange(h)[::-1]
    x1_3d, x2_3d = np.meshgrid(x1, x2)

    # plt.clf()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    ax = plt.subplot(1, 1, 1, projection='3d')
    
    ax.plot_surface(
        x1_3d, x2_3d, data, 
        rstride=1, cstride=1, alpha=1.0, 
        cmap=cm.get_cmap('jet'))
    
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    ax.view_init(55.81, -28.06)
    
    plt.show()

    # to find viewpoint
    print()
    print('ax.elev : {}'.format(ax.elev))
    print('ax.azim : {}'.format(ax.azim))
    print()
