#import torch
#import torch.utils.data as data
from fid import calculate_fid_given_paths
#import torchvision.transforms as transforms
from inception_score.model import get_inception_score
from PIL import Image
import pathlib
import argparse
import os
import numpy as np
import random

parser = argparse.ArgumentParser(description='Eval modeling ...')
# experiment
#parser.add_argument('--exp-index', type=int, default=2)
parser.add_argument('--gen_image_path', type=str, default='')
parser.add_argument('--gen_mode', type=str,default='m0')
parser.add_argument('--use_gpus', type=str, default='')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

np.random.seed(2019)
random.seed(2019)

real_path = 'data/raw_test_image/'
def FID(batch_size=32, cuda=True, dims=2048, gen_mode=''):
	gen_path = args.gen_image_path

	paths = [real_path, gen_path]

	return calculate_fid_given_paths(paths, inception_path=None,low_profile=False, gen_mode=args.gen_mode)

def IS_SCORE(gen_mode=args.gen_mode):
	path = args.gen_image_path
	path = pathlib.Path(path)	
	all_images = list(path.glob(gen_mode+'*.jpg')) + list(path.glob(gen_mode+'*.png'))

	images_array = []
	for img in all_images:
		image = Image.open(img)
		images_array.append(np.array(image).astype(np.float32))
	
	is_score, w = get_inception_score(images_array)
	
	return is_score

if __name__ == '__main__':
	'''print('calcualte the fid score ...')
	fid = FID(gen_mode=args.gen_mode)
	print('done')
	print('FID Distance', fid)
	'''	
	print('calculate inception score ...')
	is_score = IS_SCORE()
	print('done')

	print('IS Score: ', is_score)
		
