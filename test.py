import torch
import torchvision

from dataloader import data_loader
from model.model import Generator
from utils import make_img, make_z

import os
import numpy as np
import argparse


def main(args):    
    dloader, dlen = data_loader(root=args.root, batch_size=30, shuffle=True, 
                                img_size=128, mode='test', dstname='sketch')

    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args.epochs is not None:
        weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=args.epochs)
    else:
        weight_name = 'checkpoint_100_epoch.pkl'
        
    checkpoint = torch.load(os.path.join(args.weight_dir, weight_name))
    G = Generator(z_dim=8).type(dtype)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epochs is None:
        args.epochs = 'latest'
    img_name = '{type}_{epoch}.png'.format(type=args.sample_type, epoch=args.epochs)
    img_path = os.path.join(args.result_dir, img_name)

    # Make latent code and images
    z = make_z(n=dlen, img_num=args.img_num, z_dim=8, sample_type=args.sample_type)

    result_img = make_img(dloader, G, z, img_num=args.img_num, img_size=128)  
    torchvision.utils.save_image(result_img, img_path, nrow=args.img_num + 1, padding=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_type', type=str, default='random',
                        help='Type of sampling : \'random\' or \'interpolation\'') 
    parser.add_argument('--root', type=str, default='C:/Dataset/sketch',
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='results/test',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='weight',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--img_num', type=int, default=50,
                        help='Generated images number per one input image')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args([])
    main(args)