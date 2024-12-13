import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
import os
import time
import datetime
from dataloader import data_loader
from utils import init_torch_seeds, make_img, var
from losses import BCELoss, VGGLoss, MSELoss, KLDLoss
from model.model import Generator, Discriminator, Encoder, Latent_Discriminator

# for reproductionary
init_torch_seeds(seed=1)

class Solver():
    def __init__(self, root='dataset/anime_faces', dstname='sketch', result_dir='result', weight_dir='weight', load_weight=False,
                 batch_size=1, test_size=10, test_img_num=5, img_size=128, num_epoch=100, save_every=1000, save_epoch=5,
                 g_lr=0.0002, d_lr=0.0001, beta_1=0.5, beta_2=0.999, lambda_kl=0.01, lambda_img=10, lambda_z=0.5, \
                     z_dim=8, logdir=None, epochs=1):
        
        # Data type(Can use GPU or not?)
        self.dtype = torch.cuda.FloatTensor
        if torch.cuda.is_available() is False:
            self.dtype = torch.FloatTensor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data loader for training
        self.dloader, dlen = data_loader(root=root, batch_size=batch_size, shuffle=True, 
                                         img_size=img_size, mode='train', dstname=dstname)
        print('training dataset length:', dlen)
        # Data loader for test
        self.t_dloader, _ = data_loader(root=root, batch_size=test_size, shuffle=False, 
                                        img_size=img_size, mode='test', dstname=dstname)

        # Models
        # Di is discriminator for image
        # Dz is discriminator for latent code
        # G is generator for input image and latent code z
        # Z is encoder for information difference
        self.Di = Discriminator().type(self.dtype)
        self.Dz = Latent_Discriminator().type(self.dtype)
        self.G = Generator().type(self.dtype)
        self.E = Encoder().type(self.dtype)

        # Optimizers
        self.optim_Di = optim.Adam(self.Di.parameters(), lr=d_lr, betas=(beta_1, beta_2))
        self.optim_Dz = optim.Adam(self.Dz.parameters(), lr=d_lr, betas=(beta_1, beta_2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=g_lr, betas=(beta_1, beta_2))
        self.optim_E = optim.Adam(self.E.parameters(), lr=g_lr, betas=(beta_1, beta_2))

        # fixed random_z for test
        self.fixed_z = var(torch.randn(test_size, test_img_num, z_dim))

        # losses
        self.bce_loss = BCELoss
        self.recon_x_loss = VGGLoss()
        self.recon_z_loss = nn.L1Loss()
        self.mse_loss = MSELoss
        self.kl_loss = KLDLoss()


        # Some hyperparameters
        self.z_dim = z_dim
        self.lambda_img = lambda_img
        self.lambda_kl = lambda_kl
        self.lambda_z = lambda_z

        self.writer = SummaryWriter(logdir)

        # Extra things
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight
        self.epochs = epochs
        self.test_img_num = test_img_num
        self.img_size = img_size
        self.start_epoch = 0
        self.num_epoch = num_epoch
        self.save_every = save_every
        self.save_epoch = save_epoch
    
    '''
        < show_model >
        Print model architectures
    '''
    def show_model(self):
        print('================================ Discriminator for image =====================================')
        print(self.Di)
        print('==========================================================================================\n\n')
        print('================================ Discriminator for Latent Space =====================================')
        print(self.Dz)
        print('==========================================================================================\n\n')
        print('================================= Generator ==================================================')
        print(self.G)
        print('==========================================================================================\n\n')
        print('================================= Encoder ==================================================')
        print(self.E)
        print('==========================================================================================\n\n')
        
    '''
        < set_train_phase >
        Set training phase
    '''
    def set_train_phase(self):
        self.Di.train()
        self.Dz.train()
        self.G.train()
        self.E.train()
    
    '''
        < load_checkpoint >
        If you want to continue to train, load pretrained weight from checkpoint
    '''
    def load_checkpoint(self, checkpoint):
        print('Load model')
        self.Di.load_state_dict(checkpoint['discriminator_image_state_dict'])
        self.Dz.load_state_dict(checkpoint['discriminator_latent_state_dict'])
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.E.load_state_dict(checkpoint['encoder_state_dict'])
        self.optim_Di.load_state_dict(checkpoint['optim_di'])
        self.optim_Dz.load_state_dict(checkpoint['optim_dz'])
        self.optim_G.load_state_dict(checkpoint['optim_g'])
        self.optim_E.load_state_dict(checkpoint['optim_e'])
        self.start_epoch = checkpoint['epoch']
        
    '''
        < save_checkpoint >
        Save checkpoint
    '''
    def save_checkpoint(self, state, file_name):
        print('saving check_point')
        torch.save(state, file_name)
    
    '''
        < all_zero_grad >
        Set all optimizers' grad to zero 
    '''
    def all_zero_grad(self):
        self.optim_Di.zero_grad()
        self.optim_Dz.zero_grad()
        self.optim_G.zero_grad()
        self.optim_E.zero_grad()

    '''
        < train >
        Train the D_image, D_latnet, G and E 
    '''
    def train(self):
        if self.load_weight is True:
            weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=self.epochs)
            checkpoint = torch.load(os.path.join(self.weight_dir, weight_name))
            self.load_checkpoint(checkpoint)
        
        self.set_train_phase()
        self.show_model()

        print('====================     Training    Start... =====================')
        for epoch in range(self.start_epoch, self.num_epoch + 1):
            start_time = time.time()

            for iters, (img, ground_truth) in tqdm(enumerate(self.dloader)):
                # img : (1, 3, 128, 128) of domain A / ground_truth : (1, 3, 128, 128) of domain B
                img, ground_truth = var(img), var(ground_truth)

                # seperate data for image and z latent space
                data = {'img' : img[0].unsqueeze(dim=0), 'ground_truth' : ground_truth[0].unsqueeze(dim=0)}

                ''' ----------------------------- 1. Train D ----------------------------- '''
                # encoded latent vector
                z_hat, _, _ = self.E(data['ground_truth'])
                # generate fake image 
                x_tilde = self.G(data['img'], z_hat)

                # random latent vector
                z = var(torch.randn(1, self.z_dim))
                # generate fake image 
                x_hat = self.G(data['img'], z)
                # encoded latent vector
                z_tilde, _, _ = self.E(x_hat)

                # get scores and loss
                real_pair = torch.cat([data['img'], data['ground_truth']], dim=1)
                fake_pair_tilde = torch.cat([data['img'], x_tilde], dim=1)
                fake_pair_hat = torch.cat([data['img'], x_hat], dim=1)

                real_d = self.Di(real_pair)
                fake_d_tidle = self.Di(fake_pair_tilde.detach())
                fake_d_hat = self.Di(fake_pair_hat.detach())

                real_z = self.Dz(z)
                fake_z_hat = self.Dz(z_hat)
                fake_z_tidle = self.Dz(z_tilde)

                loss_images = (self.mse_loss(real_d, target=1) * 2 + self.mse_loss(fake_d_tidle, target=0) + \
                                    self.mse_loss(fake_d_hat, target=0)) / 4
                loss_latent = (self.bce_loss(real_z, target=1) * 2 + self.bce_loss(fake_z_hat, target=0) + \
                                    self.bce_loss(fake_z_tidle, target=0)) / 4
                
                d_loss = loss_images + loss_latent

                self.writer.add_scalars('d_losses', {'images_loss': loss_images, 'latent_loss': loss_latent}, epoch)

                # Update D
                self.all_zero_grad()
                d_loss.backward()
                self.optim_Di.step()
                self.optim_Dz.step()
                

                ''' ----------------------------- 2. Train G & E ----------------------------- '''
                # encoded latent vector
                mu, log_variance, z_hat = self.E(data['ground_truth'])
                # generate fake image 
                x_tilde = self.G(data['img'], z_hat)

                # random latent vector
                z = var(torch.randn(1, self.z_dim))
                # generate fake image 
                x_hat = self.G(data['img'], z)
                # encoded latent vector
                z_tilde, _, _ = self.E(x_hat)

                # get scores and loss
                fake_pair_tilde = torch.cat([data['img'], x_tilde], dim=1)
                fake_pair_hat = torch.cat([data['img'], x_hat], dim=1)

                fake_d_tidle = self.Di(fake_pair_tilde)
                fake_d_hat = self.Di(fake_pair_hat)

                fake_z_hat = self.Dz(z_hat)
                fake_z_tidle = self.Dz(z_tilde)

                loss_images = (self.mse_loss(fake_d_tidle, target=1) +  self.mse_loss(fake_d_hat, target=1)) / 2
                loss_latent = (self.bce_loss(fake_z_hat, target=1) + self.bce_loss(fake_z_tidle, target=1)) / 2
                g_loss = loss_images + loss_latent
                loss_x_recon = self.recon_x_loss(x_tilde, data['ground_truth']) * self.lambda_img
                loss_z_recon = self.recon_z_loss(z, z_tilde) * self.lambda_z
                loss_kl = self.lambda_kl * self.kl_loss(mu, log_variance)

                eg_loss = g_loss + loss_x_recon + loss_z_recon + loss_kl

                self.all_zero_grad()
                eg_loss.backward()
                self.optim_E.step()
                self.optim_G.step()

                self.writer.add_scalars('eg_losses', {'images_loss': loss_images, 'latent_loss': loss_latent, 'x_recon_loss': loss_x_recon, \
                    'z_recon_loss': loss_z_recon, 'kl_loss': loss_kl,}, epoch)

                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))
                
                # Print error and save intermediate result image and weight
                if iters % self.save_every == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('[Elapsed : %s / Epoch : %d / Iters : %d] => D_loss : %f / G_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f'\
                          %(et, epoch, iters, d_loss.item(), g_loss.item(), loss_kl.item(), loss_x_recon.item(), loss_z_recon.item()))

                    # Save intermediate result image
                    if os.path.exists(self.result_dir) is False:
                        os.makedirs(self.result_dir)

                    self.G.eval()
                    with torch.no_grad():
                        result_img = make_img(self.t_dloader, self.G, self.fixed_z, 
                                                img_num=self.test_img_num, img_size=self.img_size)

                    img_name = '{epoch}_{iters}.png'.format(epoch=epoch, iters=iters)
                    img_path = os.path.join(self.result_dir, img_name)

                    torchvision.utils.save_image(result_img, img_path, nrow=self.test_img_num+1)

                    # Save intermediate weight
                    if os.path.exists(self.weight_dir) is False:
                        os.makedirs(self.weight_dir)
                    
            
            # Save weight at the end of every epoch
            if epoch % self.save_epoch == 0:
                # self.save_weight(epoch=epoch)
                checkpoint = {
                    "generator_state_dict": self.G.state_dict(),
                    "discriminator_image_state_dict": self.Di.state_dict(),
                    "discriminator_latent_state_dict": self.Dz.state_dict(),
                    "encoder_state_dict": self.E.state_dict(),
                    "optim_g": self.optim_G.state_dict(),
                    "optim_di": self.optim_Di.state_dict(),
                    "optim_dz": self.optim_Dz.state_dict(),
                    "optim_g": self.optim_G.state_dict(),
                    "optim_e": self.optim_E.state_dict(),
                    "epoch": epoch
                    }
                path_checkpoint = os.path.join(self.weight_dir, "checkpoint_{}_epoch.pkl".format(epoch))
                self.save_checkpoint(checkpoint, path_checkpoint)

            self.writer.close()
            print("Training ending...")