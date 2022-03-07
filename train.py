#--------------------------------
#author: xxx
#code for our paper: ReGO: Reference-Guided Outpainting for Scenery Image

#Our code is built on the top of this project: https://github.com/recong/Boundless-in-Pytorch, 
#Thanks the author's contribution
#--------------------------------
import argparse
from datetime import datetime
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import *
from models import *
import time
import pdb
import time
from network import adjust_learning_rate, Preceptual_Loss
import numpy as np
import torch
from network import StyleRank_Loss

t_start = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument("--lambda_s", type=float, default=0.1)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--dataset_name", type=str, default="./data/raw_train_image_sk/", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=54, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=1e-3, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_shape", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=150, help="batch interval between model checkpoints")
parser.add_argument("--warmup_epochs", type=int, default=20, help="number of epochs with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=1e-2, help="adversarial loss weight")
parser.add_argument("--save_images", default='images', help="where to store images")
parser.add_argument("--save_models", default='saved_models', help="where to save models")
parser.add_argument("--mask_ratio", type=float, default=0.5, help="the ratio of pixels in the image are respectively masked")
parser.add_argument("--gpu", type=list, default='0', help="gpu number")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save_images):
    os.makedirs(opt.save_images)
if not os.path.exists(opt.save_models):
    os.makedirs(opt.save_models)

if torch.cuda.is_available():
    #torch.cuda.set_device(opt.gpu)
    #device = torch.device('cuda:{}'.format(opt.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu)
else:
    device = torch.device('cpu')

hr_shape = opt.hr_shape
gpus = [int(item) for item in opt.gpu]
multi_gpu = len(gpus) > 1
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator256()#.to(device)
if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
if multi_gpu:
    generator = torch.nn.DataParallel(generator, device_ids =gpus)#.to(device)
generator = generator.cuda()
if hr_shape == 256:
    ie = InceptionExtractor256()#.to(device)
    #discriminator = Discriminator256().to(device)
    if multi_gpu:
        ie = torch.nn.DataParallel(ie, device_ids=gpus)#.to(device)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=gpus)#.to(device)
    ie, discriminator = ie.cuda(), discriminator.cuda()
elif hr_shape == 512:
    ie = InceptionExtractor512().to(device)
    discriminator = Discriminator512().to(device)
else:
    print('This input shape is not available')
    exit()

# Set feature extractor to inference mode
ie.eval()

# Losses
criterion_content = torch.nn.L1Loss()#.to(device)
criterion_pixel = torch.nn.L1Loss()#.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset_name, hr_shape=hr_shape, ratio=opt.mask_ratio),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------
total_batch = len(dataloader)
data_prefetcher = Data_Prefetcher(dataloader)
precep_loss = Preceptual_Loss(opt)  
style_loss = StyleRank_Loss(opt)

for epoch in range(opt.epoch, opt.n_epochs):
    D_loss = 0
    G_loss = 0
    s_loss = 0
    content = 0
    adv = 0
    pixel = 0
    prep = 0
    count = 0
    i = 0
    imgs = data_prefetcher.fetch_next()    
    start_time = time.time()
    if (epoch+1) > 1000:
        adjust_learning_rate(opt.lr_g, optimizer_G, epoch+1)
        adjust_learning_rate(opt.lr_d, optimizer_D, epoch+1)
    #for i, imgs in enumerate(dataloader):
    while imgs != None:
        # Configure model input
        i +=1
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        img_alpha = Variable(imgs["alpha"].type(Tensor))
        clip = Variable(imgs['clip'].type(Tensor))
        left_im = clip[:,:3,:,:]
        left_im = left_im[:,:,:,:128]
        
        class_cond = ie(imgs_hr).detach()
        sketch = imgs['sketch'].cuda()
        posi_cns = imgs['posi_cn'].cuda()
        detail_ims = imgs['detail'].cuda()
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        
        # Generate a extended image from input
        gen_hr = generator(clip, sketch, posi_cns, detail_ims)
        # Measure pixel-wise loss against ground truth
        prep_loss = precep_loss(gen_hr, imgs_hr) 
        gen_hr_d = gen_hr * img_alpha + imgs_lr

        if epoch < opt.warmup_epochs:
            # Warm-up (pixel-wise loss only)
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            loss_pixel += prep_loss
            loss_pixel.backward()
            optimizer_G.step()
            pixel += loss_pixel.item()
            if i % 50 == 0:
                print('warmup!-epoch:[%d/%d],batch[%d/%d]'%(epoch, opt.warmup_epochs, i, total_batch))

            imgs = data_prefetcher.fetch_next()    
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr, img_alpha, class_cond, sketch).detach()
        pred_fake = discriminator(gen_hr_d, img_alpha, class_cond, sketch)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = -pred_fake.mean()
        styleLoss = style_loss(left_im, gen_hr[:,:,:,128:], detail_ims)

        # Total generator loss
        loss_G = opt.lambda_adv * loss_GAN + prep_loss + opt.lambda_s * styleLoss

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr, img_alpha, class_cond, sketch)
        pred_fake = discriminator(gen_hr_d.detach(), img_alpha, class_cond, sketch)

        # Total loss
        loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        D_loss += loss_D.item()
        G_loss += loss_G.item()
        s_loss += styleLoss.item()
        adv += loss_GAN.item()
        pixel += prep_loss.item()
        imgs = data_prefetcher.fetch_next()    
        
        if i%20 == 0:
            print('[%d][%d/%d], G-Loss:[%f], D-Loss:[%f], Adv-Loss:[%f], Prep-Loss:[%f], stype-loss:[%f]'%(epoch, i, total_batch, loss_G.item(), loss_D.item(), loss_GAN.item(), prep_loss.item(), styleLoss.item()))        

    avg_D_loss = D_loss / len(dataloader)
    avg_G_loss = G_loss / len(dataloader)
    avg_adv_loss = adv / len(dataloader)
    avg_pixel_loss = pixel / len(dataloader)
    avg_style_loss = s_loss / len(dataloader)
    end_time = time.time()    
    print(
        '{7}\nTime Cost: {9} mins,Epoch:{1}/{2} D_loss:{3} G_loss:{4} adv:{5} pixel:{6} time:{0}\n{8}'.format(
            datetime.now() - t_start, epoch + 1, opt.n_epochs, round(avg_D_loss,4),
            round(avg_G_loss,4), round(avg_adv_loss,4), round(avg_pixel_loss,4), '='*80, '='*80, round((end_time-start_time) / 60.0, 3)))

    if (epoch + 1) % opt.sample_interval == 0:
        # Save example results
        count += 1
        img_grid = denormalize(torch.cat((imgs_lr[:4], detail_ims[:4], gen_hr_d[:4], imgs_hr[:4]), -1))
        save_image(img_grid, opt.save_images + "/epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)
    if (epoch + 1) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        if multi_gpu:
            torch.save(generator.module.state_dict(), opt.save_models + "/generator_{}.pth".format(epoch + 1))
            torch.save(discriminator.module.state_dict(), opt.save_models + "/discriminator_{}.pth".format(epoch + 1))
        else:
            torch.save(generator.state_dict(), opt.save_models + "/generator_{}.pth".format(epoch + 1))
            torch.save(discriminator.state_dict(), opt.save_models + "/discriminator_{}.pth".format(epoch + 1))


