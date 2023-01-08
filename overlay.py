from PIL import Image, ImageEnhance
import numpy as np
import cv2 as cv
import math
from numpy.linalg import inv

def rhombize(
    img: Image,
    x_angle: float = 0,
    y_angle: float = 0,
):
    w0, h0 = img.size
    w0, h0 = w0+20, h0+20
    angles = np.radians((x_angle, y_angle))
    tanx, tany = np.tan(angles)
    cosx, cosy = np.cos(angles)

    '''
    Transform the old image coordinates to the new image coordinates
    [ a b c ][ x ]   [ x']
    [ d e f ][ y ] = [ y']
    [ 0 0 1 ][ 1 ]   [ 1 ]
    '''
    shear = np.array((
        # x col    y col      global col
        (1/cosx,   tanx, max(0, -h0*tanx)),  # -> x' row
        (  tany, 1/cosy, max(0, -w0*tany)),  # -> y' row
        (     0,      0,                1),  # -> 1  row
    ))

    size_transform = np.abs(shear[:2, :2])
    w1, h1 = ((size_transform @ img.size)).astype(int)
    '''
    The original implementation was assigning old coordinates to new
    coordinates on the left-hand side, so this needs to be inverted
    '''
    shear = inv(shear)

    rhombised = img.transform(
        size=(w1, h1),
        method=Image.AFFINE,
        data=shear[:2, :].flatten(),
        resample=Image.BILINEAR,
        fillcolor=(255,255,255,0),)

    return rhombised



def overlay(im1, im2):

    brightness = np.random.randint(low=5, high=20) / 10
    contrast = np.random.randint(low=5, high=20) / 10
    im = Image.open(im2)
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(im_output)
    im_output = enhancer.enhance(brightness)
    im_output.save('arrow.jpeg')




    img1 = Image.open(im1)
    img2 = Image.open("arrow.jpeg")

    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')

    factor = np.abs(np.random.randn(1))
    shape = (int(factor*150),int(factor*100))
    img2 = img2.resize(shape)

    location = np.random.randint(low=0, high=np.asarray(img1.size)-np.asarray(img2.size))
    angle1 = np.random.randint(low=0, high=30)
    angle2 = np.random.randint(low=0, high=30)
    angle = np.random.randint(low=-10, high=10)
    lr = np.random.randint(low=0, high=2)

    img2 = rhombize(img2, angle1, angle2)
    img2.save('arrow.png')
    img = Image.open('./arrow.png')
    img.convert('RGBA')
    img = img.rotate(angle+lr*180, fillcolor=(255,255,255,0), resample=Image.BILINEAR)

    print("Size:"+str(img.size))

    if (lr == 0):
        print("Orientation:Right")
        print("Position:" + str(np.int0((location[0] + img.size[0] / 2, location[1] + img.size[1] / 2))))
    else:
        print("Orientation:Left")
        print("Position:" + str(np.int0((location[0] - img.size[0] / 2, location[1] - img.size[1] / 2))))


    img1.paste(img, tuple(location), img)
    img1.show()
    img1.save("final.png")

    return img1

im1 = "./90278.jpg"
im2 = "./360_F_363874204_bCi6pYu30qLmGiMonDzzDAxytvDitm3K.jpg"

img = overlay(im1, im2)
