import torch
import torchvision

from dataloader import data_loader
from model.model import Generator
from utils import make_z, var

import os
import numpy as np
import argparse


def make_img_split(dloader, G, z, img_num=5):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    dloader = iter(dloader)
    img, img_real = dloader.next()

    N = img.size(0)    
    img = var(img.type(dtype))
    img_real = var(img_real.type(dtype))

    for i in range(N):
        # generate img_num images per a domain B image
        real_img = img_real[i].data / 2 + 0.5
        img_name_ = '{idx}.png'.format(idx=str(i + 1))
        real_img_path = os.path.join(args.result_dir, "ground_truth", img_name_)
        if os.path.exists(real_img_path) is False:
            os.makedirs(real_img_path)
        torchvision.utils.save_image(real_img, real_img_path)

        # Insert generated images to the next of the original image
        for j in range(img_num):
            img_ = img[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)
            
            out_img = G(img_, z_)
            out_img = out_img.data / 2 + 0.5
            img_name = '{idx}.png'.format(idx=str(i * img_num + j + 1))
            gen_img_path = os.path.join(args.result_dir, "generated", img_name)
            if os.path.exists(gen_img_path) is False:
                os.makedirs(gen_img_path)
            torchvision.utils.save_image(out_img, gen_img_path)


def main(args):    
    dloader, dlen = data_loader(root=args.root, batch_size=30, shuffle=True, 
                                img_size=128, mode='test', dstname='anime')

    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args.epochs is not None:
        weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=args.epochs)
    else:
        weight_name = 'checkpoint_1_epoch.pkl'
        
    checkpoint = torch.load(os.path.join(args.weight_dir, weight_name))
    G = Generator(z_dim=8).type(dtype)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)

    # Make latent code and images
    z = make_z(n=dlen, img_num=args.img_num, z_dim=8, sample_type=args.sample_type)

    make_img_split(dloader, G, z, img_num=args.img_num)  

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_type', type=str, default='interpolation',
                        help='Type of sampling : \'random\' or \'interpolation\'') 
    parser.add_argument('--root', type=str, default='../input/animeface', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='./',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='../input/weights',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--img_num', type=int, default=10,
                        help='Generated images number per one input image')
    parser.add_argument('--epochs', type=int, default=64,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args([])
    main(args)