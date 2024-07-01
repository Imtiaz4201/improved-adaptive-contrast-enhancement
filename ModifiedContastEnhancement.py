# Imtiaz Ahmad (2023)

import numpy as np
import cv2
import math


class ContastEnhancement(object):
    def __init__(self, image) -> None:
        self.image = image # image path

    def calculate_gamma(self, img, tau=3) -> float:
        """
        Rahman, Shanto & Rahman, Md. Mostafijur & Abdullah-Al-Wadud, M. & Al-Quaderi, Golam Dastegir & Shoyaib, Mohammad. (2016). 
        An adaptive gamma correction for image enhancement. EURASIP Journal on Image and Video Processing. 35. 10.1186/s13640-016-0138-1. 
        """
        """
            Calculate gamma of the image based on V(brightness)
        """
        def heaviside(x):
            if x >= 0:
                return 1
            else:
                return 0

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

        # Determine contrast class
        if (4*std) <= (1/tau):
            class_ = "lc"
        else:
            class_ = "hc"
        # Determine whether the image is dark or bright
        if mean >= 0.50:
            sub_class_ = class_ + 'b'
        else:
            sub_class_ = class_ + 'd'
        # Calculate gamma based on contrast and brightness
        if sub_class_ == 'lcb' or sub_class_ == 'lcd':  # for all type of low contrast
            y = -math.log(std, 2)
        else:
            # for all type of high contrast
            y = math.exp((1 - (mean + std)) / 2)

        gamma = 0
        y_ = y
        K = y_ + (1-y_) * (mean**y)
        c = 1/(1+heaviside(0.5-mean)*(K-1))
        
        # Re-Calculate gamma based on sub classes of image's contrast and brightness
        # Bright images in low contrast
        if sub_class_ == 'lcb' and mean >= 0.5:
            gamma = y_**(1-mean)
        elif sub_class_ == "lcd" and mean < 0.5:
            gamma = y_ / (y_ + (1-y_) * (mean**y))
        else:
            # hcd or hcb
            # gamma = (c*y_)
            if sub_class_ == "hcd":
                gamma = (c*std)**y_
            else:
                # send c and y to enhance hsv in agcwd (only for hcb)
                return (c, y)
        return gamma

    def agcwd(self, image, WF) -> list:
        """
        S. -C. Huang, F. -C. Cheng and Y. -S. Chiu, 
        "Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041, March 2013, doi: 10.1109/TIP.2012.2226047.
        """
        hsv_img = None

        if type(WF) != tuple:
            normalized_image = image.astype(np.float32) / 255.0
            bgr = normalized_image.copy()
            bgr /= 255.0
            hsv_img = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv_img)
            V_normalized = V * 255.0  # normalized intensity
            pdf_ = self.calculate_pdf(V_normalized)
            cdf_ = self.calculate_cdf(pdf_, WF)

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
        else:
            """
             Rahman, Shanto & Rahman, Md. Mostafijur & Abdullah-Al-Wadud, M. & Al-Quaderi, Golam Dastegir & Shoyaib, Mohammad. (2016). 
             An adaptive gamma correction for image enhancement. EURASIP Journal on Image and Video Processing. 35. 10.1186/s13640-016-0138-1. 
            """
            '''
            -The image is bright and having high contrast.
            -Equations 2, 4 and 9 are used from 'An adaptive gamma correction for image enhancement'
            '''
            c, y = WF
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            _, _, V = cv2.split(hsv_img)
            v_normalized = V.astype(np.float64) / 255.0
            i_out = c*(v_normalized**y)
            i_out = i_out * 255
            hsv_img[:, :, 2] = i_out
            bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
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
        gamma = self.calculate_gamma(image, 3)

        processed_image = self.agcwd(image, gamma)
        return processed_image
