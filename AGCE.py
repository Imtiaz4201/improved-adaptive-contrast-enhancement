# Imtiaz Ahmad (2023)
'''
research paper:
Rahman, Shanto & Rahman, Md. Mostafijur & Abdullah-Al-Wadud, M. & Al-Quaderi, Golam Dastegir & Shoyaib, Mohammad. (2016). 
An adaptive gamma correction for image enhancement. EURASIP Journal on Image and Video Processing. 35. 10.1186/s13640-016-0138-1. 
'''

import cv2
import math
import numpy as np


def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0


def adaptive_gamma_correction(img, tau=3):
    """
        Calculate gamma of the image based on V(brightness)
    """
    # convert to HSV
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split HSV
    H, S, V = cv2.split(HSV)
    # normalzie V/brightness
    v_normalized = V.astype(np.float64) / 255.0
    mean, std = cv2.meanStdDev(v_normalized)
    mean = mean[0][0]
    std = std[0][0]
    sub_class_ = None
    class_ = None

    # find out contrast
    if (4*std) <= 1/tau:
        class_ = "lc"
    else:
        class_ = "hc"

    # find out whether image is dark or bright
    if mean >= 0.5:
        sub_class_ = class_ + 'b'
    else:
        sub_class_ = class_ + 'd'

    if sub_class_ == 'lcb' or sub_class_ == 'lcd':  # for all type of low contrast
        y = -math.log(std, 2)
    else:
        y = math.exp((1 - (mean + std)) / 2)  # for all type of high contrast

    # --------
    i_in = v_normalized
    i_out = i_in
    i_in_power_of_y = (i_in**y)
    K = i_in_power_of_y + (1-i_in_power_of_y) * (mean**y)
    c = 1/(1+heaviside(0.5-mean)*(K-1))

    # Bright images in low contrast
    if sub_class_ == 'lcb' and mean >= 0.5:
        i_out = c*i_in_power_of_y
    elif sub_class_ == "lcd" and mean < 0.5:
        i_out = i_in_power_of_y / \
            (i_in_power_of_y + (1-i_in_power_of_y) * (mean**y))
    else:
        # hcd or hcb
        i_out = c*i_in_power_of_y

    i_out = i_out * 255
    HSV[:, :, 2] = i_out
    rgb = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
    return rgb
