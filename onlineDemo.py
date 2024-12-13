import torch
import torchvision
import gradio as gr
from torchvision import transforms
from model.model import Generator
from utils import make_z


transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                        std=(0.5, 0.5, 0.5))
                                   ])

def diverse_colorization(image):
    # Make latent code and images
    image = transform(image)
    z = make_z(n=1, img_num=6, z_dim=8, sample_type='random')

    # load pretrained model
    checkpoint = torch.load('weight/checkpoint_100_epoch.pkl')
    G = Generator(z_dim=8).cuda()
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()

    # colorize image
    input_img = torch.FloatTensor(image).cuda()
    result_img = torch.FloatTensor(7, 3, 128, 128).cuda()
    
    # original image to the leftmost
    result_img[0] = input_img.data
    # Insert generated images to the next of the original image
    for i in range(6):
        img_ = input_img.unsqueeze(dim=0)
        z_ = z[0, i, :].unsqueeze(dim=0)
        out_img = G(img_, z_)
        result_img[0 + i + 1] = out_img.data

    # [-1, 1] -> [0, 1]
    result_img = result_img / 2 + 0.5
    # output
    torchvision.utils.save_image(result_img, 'output.png', nrow=7, padding=4)

    return 'output.png'


if __name__ == '__main__':
    interface = gr.Interface(diverse_colorization, inputs=gr.inputs.Image(type='pil'), outputs='image', title='动漫线稿多样性着色', description='请在左边的输入框中输入一张动漫线稿图像：')
    interface.launch(share=True)