import argparse
import os
from PIL import Image
import numpy as np

import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import glob
import random


from datasets import *
from models import *
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--image-path", default='../Boundless/data_detail/detail_img/')
parser.add_argument("--base-path", default='../Boundless/data/raw_train_image_sk/')
parser.add_argument("--cand-path", default='../Boundless/data_detail/edit_cands.json')
parser.add_argument("--ref-path", default='../Boundless/data_detail/gen_1000/')
parser.add_argument("--rand-test", action='store_true')

parser.add_argument("--output", default='./edit_imgs/', help="where to save output")
parser.add_argument("--model", default="saved_models/generator_800.pth", help="generator model pass")
parser.add_argument("--mask_ratio", type=float, default=0.5, help="the ratio of pixels in the image are respectively masked")
parser.add_argument("--gpu", type=int, default=1, help="gpu number")
parser.add_argument("--use-gpu", action='store_true')
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

all_imgs = os.listdir(opt.image_path)
all_images = list(filter(lambda x: x.startswith('im'), all_imgs))
all_sketchs = [item.replace('im_', 'sk_') for item in all_images]
im_cands = json.load(open(opt.cand_path, 'r'))

all_candidates = glob.glob(opt.base_path + '/im*') 
for i in trange(len(all_images)):
        # Prepare input
        img_pth = os.path.join(opt.image_path, all_images[i])
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
        sk_pth = os.path.join(opt.image_path, all_sketchs[i])
        sk = Image.open(sk_pth)#.resize((256, 256))
        sk_arr = np.asarray(sk).astype("f") / 255.0

        sk_arr = np.where(sk_arr>=0.6, 1, 0)
        sk_tensor = torch.tensor(sk_arr)
        sk_tensor = sk_tensor.float()
        sk_tensor = sk_tensor.unsqueeze(0).to(device)
        sk_tensor = sk_tensor.unsqueeze(1)

        cand_list = im_cands[all_images[i]] 
        
        #prepare the reference images
        ref_pth = os.path.join(opt.ref_path, all_images[i].replace('.jpg','.png'))
        ref = Image.open(ref_pth)#.resize((256, 256))
        # summary(generator, (4, height, width))
        ref = np.asarray(ref).astype("f").transpose(2, 0, 1) / 127.5 - 1.0
        ref = torch.from_numpy(ref).to(device)
        ref = ref.unsqueeze(0)
     
        # Calculate image
        with torch.no_grad():
            for k in range(len(cand_list)):
                    cand = cand_list[k]
                    if opt.rand_test:
                        rand = random.randint(0,5039)
                        detail_img = Image.open(all_candidates[rand]).convert('RGB').resize((256, 128))
                    else:
                        detail_img = Image.open(os.path.join(opt.base_path, cand)).convert('RGB').resize((256,128)) 

                    detail_img = np.asarray(detail_img).astype("f").transpose(2, 0, 1) / 127.5 - 1.0
                    detail_img = torch.from_numpy(detail_img[:,:, 128:]).unsqueeze(0).to(device)
                    gen = generator(image_tensor, sk_tensor, posi_cns, detail_img)
                    gen_f = gen * mask + img * (1 - mask)

                    # save_image((img + 1) * 0.5, opt.output + f"/raw.png")
                    # save_image((gen + 1) * 0.5, opt.output + f"/gen.png")
                    img_grid = denormalize(torch.cat((img.unsqueeze(0)[:,:,:,:128],\
                        sk_tensor[:,:,:,128:].repeat(1,3,1,1), detail_img,ref,gen_f), -1))
                    save_image(img_grid, opt.output + "/{0}_{1}.png".format(all_images[i].replace('.jpg', ''),k), nrow=1, normalize=False)
                    #save_image((gen_f + 1) * 0.5, opt.output + f"/gen_{i}.png")
