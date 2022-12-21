"""
reference: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
"""
import torch
import numpy as np
import cv2
from sklearn.metrics import normalized_mutual_info_score


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        img1 = img1.float()
        img2 = img2.float()
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / (torch.sqrt(mse)+1e-8))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    def __call__(self, img1, img2):
        img1 = img1.numpy()
        img2 = img2.numpy()
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def cal_psnr(img1, img2):
    """

    :param img1: 0-255
    :param img2: 0-255
    :return:
    """
    return PSNR()(img1, img2)


def cal_ssim(img1, img2):
    """

    :param img1: 0-255
    :param img2: 0-255
    :return:
    """
    return SSIM()(img1,img2)


def easy_psnr_ssim(img1, img2):
    """

    :param img1: 0-1, n*c*h*w, torch.tensor, float32
    :param img2: same as img1
    :return: psnr and ssim
    """

    img1 = img1.permute([0, 2, 3, 1])*255
    img2 = img2.permute([0, 2, 3, 1])*255
    psnrs, ssims = [], []
    for i in range(img1.shape[0]):
        p = cal_psnr(img1[i], img2[i])
        s = cal_ssim(img1[i], img2[i])
        psnrs.append(p)
        ssims.append(s)
    return psnrs, ssims


def easy_psnr(img1, img2):
    """

    :param img1: 0-1, n*c*h*w, torch.tensor, float32
    :param img2: same as img1
    :return: psnr and ssim
    """
    img1 = img1.permute([0, 2, 3, 1])*255
    img2 = img2.permute([0, 2, 3, 1])*255
    p = PSNR()(img1, img2)
    return p


def easy_psnr_ssim_nmi(predict, gt):
    """
    :param img1: 0-1, n*c*h*w, torch.tensor, float32
    :param img2: same as img1
    :return: psnr and ssim
    """

    img1 = img1.permute([0, 2, 3, 1])*255
    img2 = img2.permute([0, 2, 3, 1])*255
    psnrs, ssims, nmis = [], [], []
    for i in range(img1.shape[0]):
        p = cal_psnr(predict[i], gt[i])
        s = cal_ssim(predict[i], gt[i])
        n = normalized_mutual_info_score(gt[i], predict[i])
        psnrs.append(p)
        ssims.append(s)
        nmis.append(n)
    return psnrs, ssims, nmis

def easy_nmi(predict, gt):
    """
    C*H*W, int (0-255)
    """
    predict = predict.reshape(-1)
    gt = gt.reshape(-1)
    v = normalized_mutual_info_score(predict, gt)
    return v



if __name__ == '__main__':
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    x1 = torch.randint(low=0, high=255, size=(256, 256, 3))
    y1 = torch.randint(low=0, high=255, size=(256, 256, 3))

    psnr = PSNR()
    print('My PSNR:', psnr(x1, y1))
    ssim = SSIM()
    print('MY SSIM:', ssim(x1,y1))
    _x1 = x1.permute(2, 0, 1)[None].float()/255
    _y1 = y1.permute(2, 0, 1)[None].float()/255
    _psnr, _ssim = easy_psnr_ssim(_x1, _y1)
    print(easy_psnr(_x1, _y1))
    print('Easy:', _psnr, _ssim)
    x1 = x1.numpy().astype('uint8')
    y1 = y1.numpy().astype('uint8')
    psnr = peak_signal_noise_ratio(x1, y1)
    print('skimage PSNR:', psnr)
    ssim = structural_similarity(x1, y1, multichannel=True)
    print('skimage SSIM:', ssim)
