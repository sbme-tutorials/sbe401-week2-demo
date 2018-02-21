# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images_files = [ join("./images" , f) for f in listdir("images") if isfile(join("images" , f)) ]

images = [ mpimg.imread( f ) for f in images_files ]

# https://stackoverflow.com/a/12201744
def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])


def binarize( gray_image , threshold ):
    threshold = np.max( gray_image ) * threshold
    return 1 * ( gray_image > threshold )

def view( image ):
    plt.figure(figsize=(10,20))
    plt.imshow( image )

def multi_view( images ):
    images_count = len( images )
    fig = plt.figure(figsize=(10,20))
    for row in range( images_count  ):
        ax1 = fig.add_subplot( images_count , 1 , row + 1)    
        ax1.imshow( images[ row ] )
    
gray_images = [ rgb2gray( img ) for img in images ]


binary_images = [ binarize( gray_img , 0.5 ) for gray_img in gray_images ]



combinations = list( zip( images , gray_images , binary_images ))

#[ multi_view( comb ) for comb in combinations ]

[ print( np.mean( img[...,0] )) for img in images ]
[ print( np.std( img )) for img in images ]


gray_image = gray_images[2]
view( images[2] )

x0 = 0
x1 = gray_image.shape[0] - 1
y0 = 0
y1 = gray_image.shape[1] - 1

x, y = np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)
profile = gray_image[x.astype(np.int), y.astype(np.int)]

#-- Plot...
fig, axes = plt.subplots(nrows=2)
axes[0].imshow(images[2])
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].axis('image')

axes[1].plot(profile)

plt.show()



gray_image_noisy = gray_image + np.random.rand( gray_image.shape[0] , gray_image.shape[1] ) * 255
view( gray_image_noisy )


