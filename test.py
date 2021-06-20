import argparse
import os
from PIL import Image
import numpy as np

import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import json

from models import *
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--image-path", default='../Boundless/data/raw_test_image_sk/')
parser.add_argument("--base-path", default='../Boundless/data/raw_train_image_sk/')
parser.add_argument("--cand-path", default='../Boundless/data/test_cands.json')
parser.add_argument("--output", default='gen_imgs', help="where to save output")
parser.add_argument("--model", default="saved_models/generator_400.pth", help="generator model pass")
parser.add_argument("--mask_ratio", type=float, default=0.5, help="the ratio of pixels in the image are respectively masked")
parser.add_argument("--gpu", type=int, default=1, help="gpu number")
parser.add_argument("--use_gpu", action="store_true", default=False)
parser.add_argument("--no-right", action="store_true", default=False)
opt = parser.parse_args()

os.makedirs(opt.output, exist_ok=True)
if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')
if not opt.use_gpu:
    device = torch.device('cpu')

# Define model and load model checkpoint
generator = Generator().to(device)
generator.load_state_dict(torch.load(opt.model, map_location=device))
generator.eval()

#read all images
number = len(os.listdir(opt.image_path)) // 2 + 1
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
posi_cns = posi_tensor.squeeze()
posi_cns = posi_cns.unsqueeze(0).float().to(device)

ids = list(range(number))
all_images = [opt.image_path + 'im_{}.jpg'.format(item) for item in ids]
all_sketchs = [opt.image_path + 'sk_{}.jpg'.format(item) for item in ids]
all_cands = json.load(open(opt.cand_path, 'r'))
for i in trange(len(all_images)):
        # Prepare input
        img_pth = all_images[i]
        if not os.path.exists(img_pth):
            continue 
        img = Image.open(img_pth)#.resize((256, 256))
        width, height = img.size
        # summary(generator, (4, height, width))
        edge = int(width * opt.mask_ratio)
        img = np.asarray(img).astype("f").transpose(2, 0, 1) / 127.5 - 1.0
        mask = np.zeros((height, width))  # make mask
        mask[:, edge:] = 1
        mask = mask[np.newaxis, :, :]
        img_masked = img * (1 - mask)  # apply mask
        img = torch.from_numpy(img).to(device)

        mask = torch.from_numpy(mask)
        mask_one = torch.ones((height, width), dtype=torch.float64)
        img_masked = torch.from_numpy(img_masked)
        clip = torch.cat([img_masked, mask_one[None, :, :], mask]).float()
        #clip = torch.cat([img_masked,  mask]).float()
        mask = mask.float().to(device)

        image_tensor = Variable(clip).to(device).unsqueeze(0)

        #read sketch
        sk_pth = all_sketchs[i]
        sk = Image.open(sk_pth)#.resize((256, 256))
        sk_arr = np.asarray(sk).astype("f") / 255.0

        sk_arr = np.where(sk_arr>=0.6, 1, 0)
        sk_tensor = torch.tensor(sk_arr)
        sk_tensor = sk_tensor.float()
        sk_tensor = sk_tensor.unsqueeze(0).to(device)
        sk_tensor = sk_tensor.unsqueeze(1)
        if opt.no_right:
            sk_tensor[:,:,:,128:] = 0 
        
        #read the reference images         
        detail_img_pth = all_cands[all_images[i].split('/')[-1]][0]
        detail_img = Image.open(os.path.join(opt.base_path, detail_img_pth)).resize((256, 128))
        # summary(generator, (4, height, width))
        detail_img = np.asarray(detail_img).astype("f").transpose(2, 0, 1) / 127.5 - 1.0
        detail_img = torch.from_numpy(detail_img[:,:,128:]).to(device)
        detail_img = detail_img.unsqueeze(0)
        # Calculate image
        with torch.no_grad():
            gen = generator(image_tensor, sk_tensor, posi_cns, detail_img)
            gen_f = gen * mask + img * (1 - mask)

            # save_image((img + 1) * 0.5, opt.output + f"/raw.png")
            # save_image((gen + 1) * 0.5, opt.output + f"/gen.png")
            save_image((gen_f + 1) * 0.5, opt.output + f"/gen_{i}.png")
            #print(f'Output generated image gen_{i}')
