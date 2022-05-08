from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--numImages', type=int,
                    default=100, help='number of images to be generated')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ratio', type=int, default=50,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--outf', default='/workspace/data/output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# Define all the necessary variables
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
numImages=int(opt.numImages)
nz = int(opt.nz)
ratio= int(opt.ratio)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
img_channels = 1
image_size = (256, 256)
output_dir = opt.outf

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.noiseRatio=int(nz*ratio/100)
        self.ylabel = nn.Sequential(
            nn.Linear(15, self.noiseRatio),
            nn.ReLU(True)
        )

        self.yz = nn.Sequential(
            nn.Linear(nz, nz-self.noiseRatio),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input, label):
        if input.is_cuda and self.ngpu > 1:
            # mapping noise and label
            z = nn.parallel.data_parallel(self.yz, input, range(self.ngpu))
            y = nn.parallel.data_parallel(self.ylabel, label, range(self.ngpu))

            # mapping concatenated input to the main generator network
            inp = torch.cat([z, y], 1)
            inp = inp.view(-1, nz, 1, 1)
            output = nn.parallel.data_parallel(
                self.main, inp, range(self.ngpu))
        else:
            # mapping noise and label
            z = self.yz(input)
            y = self.ylabel(label)

            # mapping concatenated input to the main generator network
            inp = torch.cat([z, y], 1)
            inp = inp.view(-1, nz, 1, 1)
            output = self.main(inp)
        return output


# custom weights initialization called on netG and netD
def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    noise = torch.randn(numImages, nz, device=device)
    with torch.no_grad():
        for cls in range(15):
            labels=torch.zeros(numImages,15)
            for i in range(numImages):
                labels[i][cls]+=1
                labels=labels.to(device)
            fake = netG(noise,labels)
            fake = fake.detach()
            for idx in range(numImages):
                vutils.save_image(fake[idx],'%s/fake%02d/i%02d_%02d.png' % (opt.outf,cls,cls,idx),normalize=True)

if __name__ == '__main__':
    main()

