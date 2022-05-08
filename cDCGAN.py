from __future__ import print_function
import argparse
import os
import gc
import random
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ratio', type=int, default=50,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.00001,
                    help='weight_decay for adam. default=1e-5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true',
                    help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--outf', default='/workspace/data/output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom',
                    help='comma separated list of classes for the lsun data set')

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
nz = int(opt.nz)
ratio= int(opt.ratio)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
epochs = int(opt.niter)
img_channels = 1
image_size = (256, 256)
output_dir = opt.outf
img_dir = '/workspace/data/NIH/images'  # TODO : Change this before running
csv_loc= '/workspace/data/NIH/mod.csv'

class customNIH(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        image_path = self.image_dir + '/' + self.df.name[index]
        image = PIL.Image.open(image_path).convert('L')
        label = torch.zeros(15)
        label[self.df.category[index]]+=1

        if self.transform is not None:
            image = self.transform(image)

        return image, label


transformation = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)]
    )
])


# custom weights initialization called on netG and netD
def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


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


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.ylabel = nn.Sequential(
            nn.Linear(15, 256*256*1),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(img_channels+1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        if input.is_cuda and self.ngpu > 1:
            y = nn.parallel.data_parallel(
                self.ylabel, labels, range(self.ngpu))
            y = y.view(-1, 1, 256, 256)
            inp = torch.cat([input, y], 1)
            output = nn.parallel.data_parallel(
                self.main, inp, range(self.ngpu))
        else:
            y = self.ylabel(labels)
            y = y.view(-1, 1, 256, 256)
            inp = torch.cat([input, y], 1)
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)


def main():
    # Create the Dataset
    df = pd.read_csv(csv_loc)
    dataset = customNIH(df, img_dir, transformation)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    # Visualize random batch of dataset and save the image
    img = next(iter(dataloader))[0]
    print("The shape of the images is :", img.shape)
    img = vutils.make_grid(img, nrow=16)

    vutils.save_image(img, '%s/random_image_grid.png' % (opt.outf))

    # Measure Time
    import time
    start_time = time.time()

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(
        opt.beta1, 0.999), weight_decay=opt.decay)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(
        opt.beta1, 0.999), weight_decay=opt.decay)
    if opt.dry_run:
        opt.niter = 1

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            print("Data shape : ", data[0].shape)
            netD.zero_grad()
            real_cpu = data[0].to(device)
            labels=data[1].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=device)
            # print("Label is : ", label)
            # print("Real cpu shape : ", real_cpu.shape)
            # print("Batch size from real_cpu : ", batch_size)
            output = netD(real_cpu,labels)
            # print("Type of output of netD : ", type(output))
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise,labels)
            label.fill_(fake_label)
            output = netD(fake.detach(),labels)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake,labels)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise,labels)
                fake = fake.detach()
                vutils.save_image(fake,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

            if opt.dry_run:
                break

        if epoch+2>=opt.niter:
            # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        
        gc.collect()
    
    print("\nTotal Compute Time :",time.time()-start_time)

if __name__ == '__main__':
    main()
