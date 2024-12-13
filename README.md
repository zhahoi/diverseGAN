# diverseGAN
Use generative adversarial networks(GANs) to diverse colorize anime sketch image.(使用生成对抗网络对动漫线稿进行多样新着色）



## Installation

To successfully run this repository, you should first install the required dependencies. (The dependencies in this repository are only for testing and do not imply that other versions cannot be used.)

```bash
pip install -r requirements.txt
```



## Datasets

The anime image dataset used in this repository comes from Kaggle. You can download it from the following link: https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair.

The dataset should be placed in the following format:

![data_format](https://github.com/zhahoi/diverseGAN/tree/main/docs/data_format.png)



## Training

```bash
python train.py --root datasets/sketch --dstname sketch --batch_size 32 --img_size 128
```

You can also manually modify the parameters in `train.py` as follows:

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='C:/Dataset/sketch',
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Result images location')
    parser.add_argument('--dstname', type=str, default='sketch', 
                        help='Choosed dataset name(sketch2color/black2color)')
    parser.add_argument('--weight_dir', type=str, default='weight', 
                        help='Weight location')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--test_size', type=int, default=8, 
                        help='Test batch size')
    parser.add_argument('--test_img_num', type=int, default=8, 
                        help='How many images do you want to generate?')
    parser.add_argument('--img_size', type=int, default=128, 
                        help='Image size')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                        help='Discriminator Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, 
                        help='Beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.999, 
                        help='Beta2 for Adam')
    parser.add_argument('--lambda_kl', type=float, default=1e-2, 
                        help='Lambda for KL Divergence')
    parser.add_argument('--lambda_img', type=float, default=10, 
                        help='Lambda for image reconstruction')
    parser.add_argument('--lambda_z', type=float, default=0.5, 
                        help='Lambda for z reconstruction')
    parser.add_argument('--z_dim', type=int, default=8, 
                        help='Dimension of z')
    parser.add_argument('--num_epoch', type=int, default=100, 
                        help='Number of epoch')
    parser.add_argument('--save_every', type=int, default=200,
                        help='How often do you want to see the result?')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='How often do you want to save the weight?')
    parser.add_argument('--load_weight', type=bool, default=False,
                        help='Load weight or not')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Epoch that you want to load the weight. ')

    args = parser.parse_args([])
    main(args=args)
```



## Test

```bash
python test.py --root datasets/sketch --sample_type random ---img_num 20 --epochs 100
```

You can also manually modify the parameters in `test.py` as follows:

```python
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
```



## Demo

You can run `onlineDemo.py` to open a UI where you can upload the sketch image that you want to colorize, and you will visually see the results.

You need to open the following URL in your browser:

![img1](https://github.com/zhahoi/diverseGAN/tree/main/docs/img1.png)

![img2](https://github.com/zhahoi/diverseGAN/tree/main/docs/img2.png)

## Reference Results

![random_100_0](https://github.com/zhahoi/diverseGAN/tree/main/results/test/random_100_0.png)

![image (1)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (1).png)

![image (2)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (2).png)

![image (3)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (3).png)

![image (4)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (4).png)

![image (5)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (5).png)

![image (6)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (6).png)

![image (7)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (7).png)

![image (8)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (8).png)

![image (9)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (9).png)

![image (10)](https://github.com/zhahoi/diverseGAN/tree/main/docs/image (10).png)

![image](https://github.com/zhahoi/diverseGAN/tree/main/docs/image.png)





## Reference

-[Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet](https://github.com/pradeeplam/Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet)

-[BicycleGAN-pytorch](https://github.com/eveningglow/BicycleGAN-pytorch)

-[AEGAN-PyTorch](https://github.com/RileyLazarou/AEGAN-PyTorch)
