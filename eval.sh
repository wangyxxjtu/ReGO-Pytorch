use_gpus=0
gen_mode='gen'

CUDA_VISIBLE_DEVICES=$use_gpus python test.py --model saved_models/generator_1500.pth --image-path ../Boundless/data/raw_test_image_sk/ --output ./gen_imgs/  --gpu 0 --use_gpu

python fid.py ~/image_outpainting/NSIO_Gate_allSketch_simple/data/raw_test_image/ ./gen_imgs/ --gen_mode $gen_mode --gpu $use_gpus

CUDA_VISIBLE_DEVICES=$use_gpus python eval_fid_is_score.py --gen_image_path ./gen_imgs/ --use_gpus $use_gpus --gen_mode $gen_mode 
