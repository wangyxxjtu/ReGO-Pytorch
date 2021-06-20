import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from prefetch_generator import BackgroundGenerator
from sketch_mask import mask_sketch_basic
import json
import sys
import pdb
import os
import random

import torchvision.transforms.functional as TF
# Normalization parameters for pre-trained PyTorch models
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

std = np.array([0.5, 0.5, 0.5])
mean = np.array([1, 1, 1])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].add_(mean[c]).mul_(std[c])
    return torch.clamp(tensors, 0, 255)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, ratio):
        self.height = 128
        self.hr_shape = hr_shape
       
        self.ratio = ratio
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std),
             ]
        )
        number = len(os.listdir(root)) // 2
        names = list(range(number))
        self.im_files = [root + '/im_{}.jpg'.format(item) for item in names]
        self.sk_files = [root + '/sk_{}.jpg'.format(item) for item in names]
        self.root = root 

        y_axis = np.array(list(range(-64, 64))).reshape((128, 1)) / 64
        y_axis = np.tile(y_axis, [1, 256])
        y_axis = np.expand_dims(y_axis, axis=-1)

        x_axis = np.array(list(range(-128, 128))).reshape((1, 256)) / 64
        x_axis = np.tile(x_axis, [128, 1])
        x_axis = np.expand_dims(x_axis, axis=-1)

        posi_arr = np.concatenate([x_axis, y_axis], axis=-1)
        posi_arr = np.expand_dims(posi_arr, axis=0)
        
        posi_tensor = torch.Tensor(posi_arr)
        posi_tensor = posi_tensor.permute(0, 3, 1, 2)
        self.posi_channels = posi_tensor.squeeze()

        #load the similar candidiate images
        self.detail_ims = json.load(open(self.root.replace('/raw_train_image_sk', '') + 'train_cands.json', 'r'))
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        print('total images: ', number)

    def __getitem__(self, index):
        hr_shape = self.hr_shape
        ratio = self.ratio
        hflip = random.random() > 0.5
        edge = int(hr_shape * ratio) + random.randint(-4, 4)
        a = Image.open(self.im_files[index % len(self.im_files)])#.resize((256, 256))
        if hflip:
            a = TF.hflip(a)
        width, height = a.size
        a = np.asarray(a).astype("f").transpose(2, 0, 1) / 127.5 - 1.0

        x = random.randint(0, height - self.height)
        y = random.randint(0, width - hr_shape)
        a = a[:, x:x + self.height, y:y + hr_shape]
        c = np.zeros((self.height, hr_shape))  # make mask
        c[:, edge:] = 1
        c = c[np.newaxis, :, :]
        d = torch.ones((self.height, hr_shape), dtype=torch.float64)
        b = a * (1 - c)  # apply mask

        c = torch.from_numpy(c)
        img_hr = torch.from_numpy(a)
        img_lr = torch.from_numpy(b)
        #print(img_lr.shape, d.shape,c.shape)
        clip = torch.cat([img_lr, d[None, :, :], c])
        #clip = torch.cat([img_lr, c])

        sk = Image.open(self.sk_files[index % len(self.sk_files)])#.resize((256, 256))
        if hflip:
            sk = TF.hflip(sk)
        sk_arr = np.asarray(sk).astype("f") / 255.0
        #pdb.set_trace()
        sk_arr = sk_arr[x:x + self.height, y:y + hr_shape]

        sk_arr = np.where(sk_arr>=0.6, 1, 0)
        sk_arr = mask_sketch_basic(np.expand_dims(sk_arr, -1))
        sk_arr = np.squeeze(sk_arr)
        #sk_arr = np.expand_dims(sk_arr, -1)
        
        sk_ten = torch.from_numpy(sk_arr)
        sk_ten = sk_ten.float()
        sk_ten = torch.unsqueeze(sk_ten, dim=0)        

        #-prepare the detail images
        detail_lst = self.detail_ims[self.im_files[index].split('/')[-1]]
        rand_idx = random.randint(0, len(detail_lst)-1)
        detail_im = Image.open(self.root + detail_lst[rand_idx])
        if hflip:
            detail_im = TF.hflip(detail_im)

        #detail_im = self.transform(detail_im)
        width, height = detail_im.size
        detail_im = np.asarray(detail_im).astype("f").transpose(2, 0, 1) / 127.5 - 1.0

        detail_im = detail_im[:, x:x + self.height, y:y + hr_shape]
        detail_tensor = torch.from_numpy(detail_im)
        shape = detail_tensor.shape
        detail_tensor = detail_tensor[:,:,shape[2]//2:] 

        img_lr, img_hr, clip, sk_ten, self.posi_channels, detail_tensor = \
        img_lr.float(), img_hr.float(), clip.float(), sk_ten.float(), \
        self.posi_channels.float(), detail_tensor.float()
        c = c.float()
            
        return {"lr": img_lr, "hr": img_hr, 'alpha': c, 'clip': clip, 'sketch':sk_ten, 'posi_cn': self.posi_channels, 'detail':detail_tensor}

    def __len__(self):
        return len(self.im_files)

class Data_Prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.dataloader = loader
        self.stream = torch.cuda.Stream()

    def fetch_next(self):
        try:
            self.imgs = next(self.loader)
        except StopIteration:
            self.imgs = None
            self.loader = iter(self.dataloader)
            return

        with torch.cuda.stream(self.stream):
            for item in self.imgs: 
                self.imgs[item] = self.imgs[item].cuda(non_blocking=True)

        return self.imgs

if __name__ == '__main__':
    #root = './data/raw_train_image_extend/'
    root = '../Boundless/data/raw_train_image_sk/'
    hr_shape = 256
    ratio = 0.5
    
    dataset = ImageDataset(root, hr_shape, ratio)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for step, image in enumerate(dataloader):
        print('ok')
        sketch = image['sketch']
        img = image['hr']
        detail = image['detail']
        Image.fromarray((sketch[0,0,:,:]*255).numpy().astype(np.uint8)).save('sk.jpg')
        img = img.permute(0, 2, 3, 1)
        Image.fromarray(((img[0]+1)*127.5).numpy().astype(np.uint8)).save('im.jpg')
        
        detail = detail.permute(0, 2, 3, 1)
        Image.fromarray(((detail[0]+1)*127.5).numpy().astype(np.uint8)).save('detail.jpg')
        pdb.set_trace()
        #sys.exit()
