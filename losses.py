import torch
import utils
from torch import nn
from model.nets import VGG19


# MSELoss for LSGAN
def MSELoss(score, target=1):
    if target == 1:
        label = utils.var(torch.ones(score.size()).fill_(0.95), requires_grad=False)
    elif target == 0:
        label = utils.var(torch.ones(score.size()).fill_(0.05), requires_grad=False)
    
    criterion = nn.MSELoss()
    loss = criterion(score, label)
    
    return loss


# BCELoss for Latent code z
def BCELoss(score, target=1):
    if target == 1:
        label = utils.var(torch.ones(score.size()).fill_(0.95), requires_grad=False)
    elif target == 0:
        label = utils.var(torch.ones(score.size()).fill_(0.05), requires_grad=False)

    criterion = nn.MSELoss()
    loss = criterion(score, label)
    
    return loss


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# wgan_loss
def WGANLoss(pred, real_or_not=True):
    if real_or_not:
        return - torch.mean(pred)
    else:
        return torch.mean(pred)


def Calculate_gradient_penalty(model, real_images, fake_images, device, constant=1.0, lamb=10.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lamb
    return gradient_penalty


