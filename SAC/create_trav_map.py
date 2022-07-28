import numpy as np
from PIL import Image

dim = 2000

def I_large(img):
    for i in range(dim):
        for j in range(dim):
            if 850 <= i < 1150:
                img[i][j] = 255

def X_large(img):
    for i in range(dim):
        for j in range(dim):
            if 850 <= i < 1150 or 850 <= j < 1150:
                img[i][j] = 255


def straight_horizontal(img):
    for i in range(dim):
        for j in range(dim):
            if 300 < i < 700:
                img[i][j] = 255


def straight_vertical(img):
    for i in range(dim):
        for j in range(dim):
            if 300 < j < 700:
                img[i][j] = 255


def straight_narrow(img):
    for i in range(dim):
        for j in range(dim):
            if 400 < i < 600:
                img[i][j] = 255


def cross(img):
    for i in range(dim):
        for j in range(dim):
            if 300 < j < 700 or 300 < i < 700:
                img[i][j] = 255


def cross_narrow(img):
    for i in range(dim):
        for j in range(dim):
            if 400 < j < 600 or 400 < i < 600:
                img[i][j] = 255


def bend(img):
    for i in range(dim):
        for j in range(dim):
            if 480 < i < 520 and 0 < j < 600:
                img[i][j] = 0


def H(img):
    for i in range(dim):
        for j in range(dim):
            if 400 < i < 600 and (0 <= j < 300 or 500 < j):
                img[i][j] = 0


def S(img):
    for i in range(dim):
        for j in range(dim):
            if (0 <= i < 600 and 575 <= j < 725) or (400 <= i and 250 <= j < 350):
                img[i][j] = 0


def O(img):
    for i in range(dim):
        for j in range(dim):
            if 300 <= i < 750 and 300 <= j < 700:
                img[i][j] = 0


def E(img):
    for i in range(dim):
        for j in range(dim):
            if (0 <= i < 800 and 650 <= j < 950) or (1300 <= i < 1700 and 450 <= j < 700) or \
                    (0 <= i < 400 and 1150 <= j < 1200) or (0 <= i < 1100 and 1450 <= j < 1500) or \
                    (700 <= i < 2000 and 1150 <= j < 1200) or (1300 <= i < 2000 and 1450 <= j < 1500) or \
                    (1500 <= i < 1700 and 1650 <= j < 1750):
                img[i][j] = 0


img = np.zeros((dim, dim), np.uint8)
#img = np.ones((dim, dim), np.uint8) * 255
X_large(img)
im = Image.fromarray(img)
im.save("floor_no_obj_0.png")
im.save("floor_trav_no_obj_0.png")
im.save("floor_trav_0_new.png")
