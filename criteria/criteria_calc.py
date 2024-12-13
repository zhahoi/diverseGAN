from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import lpips
import argparse
from criteria.make_dataset import make_dataset
import cv2
import os


def calc_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # convert image data range to [0, 255]
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1, img2 = np.array(img1), np.array(img2)
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


class util_of_lpips():
    def __init__(self, img1_path, img2_path, use_gpu=False):
        self.loss_fn = lpips.LPIPS(net='alex', version='0.1')
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()
        self.img1_path = img1_path
        self.img2_path = img2_path

    def calc_lpips(self):
        # load image
        img0 = lpips.im2tensor(lpips.load_image(self.img1_path))
        img1 = lpips.im2tensor(lpips.load_image(self.img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()

        dist01 = self.loss_fn.forward(img0, img1).squeeze(1).squeeze(1).squeeze(1).detach().numpy()
        return dist01


class Criteria():
    def __init__(self, root):
        self.root = root
        self.dataset = make_dataset(self.root)
        self.ssim_score = 0
        self.psnr_score = 0

    def train(self):
        for idx, (img_1, img_2) in enumerate(self.dataset):
            print(os.path.abspath(img_1))
            print(os.path.abspath(img_2))
      
            ssim = open('ssim_score.txt', 'a+')
            ssim_score = calc_ssim(img1_path=img_1, img2_path=img_2)
            self.ssim_score += ssim_score
            ssim.write(str(ssim_score) + '\n')

            psnr = open('psnr_score.txt', 'a+')
            psnr_score = calc_psnr(img1_path=img_1, img2_path=img_2)
            self.psnr_score += psnr_score
            psnr.write(str(psnr_score) + '\n')
            
            lpips = open('lpips_score.txt', 'a+')
            lpips_score = util_of_lpips(img1_path=img_1, img2_path=img_2).calc_lpips()
            lpips.write(str(lpips_score) + '\n')
            
        print(" ------  ssim_loss =", "{:.8f}------".format(self.ssim_score/len(self.dataset)))
        print(" ------  psnr_loss =", "{:.8f}------".format(self.psnr_score/len(self.dataset)))
    
   
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='val_data', 
                        help='Data location')
                        
    args = parser.parse_args([])
    Criteria(args.root).train()