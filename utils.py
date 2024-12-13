import random
import torch
import numpy as np

from torch.autograd import Variable
import torch.backends.cudnn as cudnn


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a
    Args:
        seed (int): The desired seed.
    """
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    print("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


'''
    < var >
    Convert tensor to Variable
'''
def var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    var = Variable(tensor.type(dtype), requires_grad=requires_grad)
    return var


'''
    < make_img >
    Generate images

    * Parameters
    dloader : Data loader for test data set
    G : Generator
    z : random_z(size = (N, img_num, z_dim))
        N : test img number / img_num : Number of images that you want to generate with one test img / z_dim : 8
    img_num : Number of images that you want to generate with one test img
'''
def make_img(dloader, G, z, img_num=5, img_size=128):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    dloader = iter(dloader)
    img, _ = dloader._next_data()

    N = img.size(0)    
    img = var(img.type(dtype))
    result_img = torch.FloatTensor(N * (img_num + 1), 3, img_size, img_size).type(dtype)

    for i in range(N):
        # original image to the leftmost
        result_img[i * (img_num + 1)] = img[i].data

        # Insert generated images to the next of the original image
        for j in range(img_num):
            img_ = img[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)
            
            out_img = G(img_, z_)
            result_img[i * (img_num + 1) + j + 1] = out_img.data

    # [-1, 1] -> [0, 1]
    result_img = result_img / 2 + 0.5
    return result_img

    
'''
    < make_interpolation >
    Make linear interpolated latent code.
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
'''
def make_interpolation(n=200, img_num=9, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    step = 1 / (img_num - 1)
    alpha = torch.from_numpy(np.arange(0, 1, step))
    interpolated_z = torch.FloatTensor(n, img_num, z_dim).type(dtype)

    for i in range(n):
        first_z = torch.randn(1, z_dim)
        last_z = torch.randn(1, z_dim)
        
        for j in range(img_num - 1):
            interpolated_z[i, j] = (1 - alpha[j]) * first_z + alpha[j] * last_z
        interpolated_z[i, img_num - 1] = last_z
    
    return interpolated_z


'''
    < make_z >
    Make latent code
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
    sample_type : random or interpolation
'''
def make_z(n, img_num, z_dim=8, sample_type='random'):
    if sample_type == 'random':
        z = var(torch.randn(n, img_num, 8))
    elif sample_type == 'interpolation':
        z = var(make_interpolation(n=n, img_num=img_num, z_dim=z_dim))
    
    return z