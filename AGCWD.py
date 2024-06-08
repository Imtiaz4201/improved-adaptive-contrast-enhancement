# Imtiaz Ahmad (2023)
"""
research paper:
S. -C. Huang, F. -C. Cheng and Y. -S. Chiu, 
"Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041, March 2013, doi: 10.1109/TIP.2012.2226047.
"""

import numpy as np
import cv2
import math


class ContastEnhancement(object):
    def __init__(self, image) -> None:
        self.image = image

    def agcwd(self, image, gamma) -> list:
        hsv_img = None
        normalized_image = image.astype(np.float32) / 255.0
        bgr = normalized_image.copy()
        bgr /= 255.0
        hsv_img = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv_img)
        V_normalized = V * 255.0  # normalized intensity
        pdf_ = self.calculate_pdf(V_normalized)
        cdf_ = self.calculate_cdf(pdf_, gamma)

        h, w = V_normalized.shape
        v = V_normalized.flatten()
        # Apply gamma correction based on the modified CDF
        '''
        T(l) = l_max * (l/l_max) ** (1-cdf(l))
        '''
        V_normalized = 255 * (v / 255.0) ** (1 - cdf_[v.astype(int)])
        V_normalized = V_normalized.reshape((h, w))

        V_normalized /= 255.0
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        bgr[:, :, 2] = V_normalized
        bgr = cv2.cvtColor(bgr, cv2.COLOR_HSV2BGR)
        bgr *= 255.0
        bgr = bgr.astype(np.uint8)
        return bgr

    def calculate_pdf(self, v) -> list:
        # Get PDF by Hist
        pdf = np.zeros((256, 1), dtype=np.float32)
        hist_size = 256
        hist_range = [0, 256]
        # Original pdf
        hist = cv2.calcHist(
            images=[v.astype(np.uint8)],
            channels=[0],
            mask=None,
            histSize=[hist_size],
            ranges=hist_range,
            accumulate=False
        )
        pdf = hist
        # NPhist, bins = np.histogram(
        #     a=v.astype(np.uint8).flatten(),
        #     bins=hist_size,
        #     range=hist_range
        # )
        # pdf = NPhist.astype(np.float32)

        # normalized pdf
        '''
        Normalizing the Probability Density Function (PDF) by dividing each value in the PDF array by 
        the total number of pixels in the image. This step ensures that the PDF values represent probabilities, 
        and the sum of all probabilities is equal to 1.
        '''
        pdf /= (v.shape[0] * v.shape[1])
        return pdf

    def calculate_cdf(self, pdf, weighting_value) -> list:
        '''
        Using the weighting distribution function to compute the CDF for gamma correction.
        '''
        max_pdf = np.max(pdf)
        min_pdf = np.min(pdf)
        cdf_ = np.zeros(pdf.shape).astype(np.float32)
        '''
            pdf(i) = pdf_max * ((pdf(i)-pdf_min)/(pdf_max-pdf_min))^alpha/α
            Where:
            alpha/α =  is the adjusted parameter/weighting_value
            pdf_max =  is the maximum of pdf
            pdf_min = is the is the minimum of pdf
        '''
        pdf_w = max_pdf * ((pdf - min_pdf) /
                           (max_pdf - min_pdf))**weighting_value
        # calculate cdf
        cdf_ = np.cumsum(pdf_w)
        #  normalizes the Cumulative Distribution Function (CDF) by dividing each value in the CDF array by the last value of the CDF. This step ensures that the CDF values are scaled between 0 and 1.
        cdf_ /= cdf_[-1]
        return cdf_

    def prcoess_image(self) -> list:
        image = self.image
        image = cv2.imread(image)
        processed_image = self.agcwd(image, 0.5)
        return processed_image
