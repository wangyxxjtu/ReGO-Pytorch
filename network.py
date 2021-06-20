import torch.nn as nn
import torch
from torchvision.models import resnet152, inception_v3
import torchvision.models as models
import pdb
from torchvision.models.resnet import model_urls
from torchvision.models.inception import model_urls as model_urls1
from places365.run_placesCNN_unified import load_model

def adjust_learning_rate(base_lr, optimizer, epoch):
    lr = base_lr * (0.1 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Conditional_Network(nn.Module):
    '''
    this is a network to encode the conditional information
    including : the image to detail the final result
    and the desird shape
    '''
    def __init__(self):
        super(Conditional_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        
        self.sk_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )
        self.sk_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.sk_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.sk_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.sk_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        
        self.p_layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )
        self.p_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.p_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.p_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.p_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
       
    def forward(self, x, sketch, posi):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        #sketch = sketch[:,:,:,128:]
        s_out1 = self.sk_layer1(sketch)
        s_out2 = self.sk_layer2(s_out1)
        s_out3 = self.sk_layer3(s_out2)
        s_out4 = self.sk_layer4(s_out3)
        s_out5 = self.sk_layer5(s_out4)
 
        #posi = posi[:,:,:,128:]
        p_out1 = self.p_layer1(posi)
        p_out2 = self.p_layer2(p_out1)
        p_out3 = self.p_layer3(p_out2)
        p_out4 = self.p_layer4(p_out3)
        p_out5 = self.p_layer5(p_out4)
       
        return out1,s_out1,p_out1, out2, s_out2, p_out2, out3, s_out3, p_out3,\
             out4,s_out4, p_out4, out5,s_out5,p_out5

#define the resblock
class ResBlock(nn.Module):
    def __init__(self,input_cns, channels):
        super(ResBlock, self).__init__()
        cn1, cn2, cn3 = channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_cns, cn1, kernel_size=1),
            nn.ELU() 
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(cn1, cn2, kernel_size=3, padding=1),
            nn.ELU()
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(cn2, cn3, kernel_size=1)
            )
           
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

#define the module to fuse the detailed right half and the left half
class Fuse_Whole(nn.Module):
    def __init__(self, input_cns, channels, rate=1):
        super(Fuse_Whole, self).__init__()
        cn1, cn2, cn3 = channels
        self.grb1_1 = nn.Sequential(
            nn.Conv2d(input_cns, input_cns, kernel_size=(3,1), padding=(rate,0), dilation=rate),
            nn.ELU()
            )
        self.grb1_2 = nn.Sequential(
            nn.Conv2d(input_cns, input_cns, kernel_size=(1, 7), padding=(0, 3*rate), dilation=rate),
            nn.ELU()
            )

        self.grb2_1 = nn.Sequential(
            nn.Conv2d(input_cns,input_cns, kernel_size=(7,1), padding=(3*rate,0), dilation=rate),
            nn.ELU()
            )
        self.grb2_2 = nn.Sequential(
            nn.Conv2d(input_cns, input_cns, kernel_size=(1, 3), padding=(0, 1*rate), dilation=rate),
            nn.ELU()
            )
    
        self.activation = nn.ELU()

        self.identity_block = ResBlock(input_cns, [input_cns//2, input_cns//2, input_cns])
        
    def forward(self, x):
        x1 = self.grb1_1(x)
        x1 = self.grb1_2(x)

        x2 = self.grb2_1(x)
        x2 = self.grb2_2(x) 
        
        x = self.activation(x1 + x2)
        x = self.identity_block(x)

        return x
        
#define the Preception Loss
class Preceptual_Loss(nn.Module):
    def __init__(self, opt):
        super(Preceptual_Loss,self).__init__()
        self.places365 = load_model()
        self.adaP = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))

        if len(opt.gpu) > 1:
            gpus = [int(item) for item in opt.gpu]
            self.places365 = torch.nn.DataParallel(self.places365, device_ids=gpus)
            self.places365.cuda()
            print('Mutiple GPUs Training')
        else:
            self.places365.cuda()
            print('Single GPUS Training')
    def forward(self, gt_im, gen_im):
        gt_fea = self.adaP(self.places365(gt_im)) 
        gen_fea = self.adaP(self.places365(gen_im))
        gt_fea = gt_fea.squeeze()
        gen_fea = gen_fea.squeeze()

        loss = torch.mean(torch.pow(gt_fea-gen_fea,2))
        return loss

class StyleRank_Loss(nn.Module):
    def __init__(self, opt):
        super(StyleRank_Loss,self).__init__()
        self.margin = opt.margin
        net = models.vgg19(pretrained=True)
        net_children = list(net.features.children())
        conv1_1 = nn.Sequential(*net_children[0:2])
        self.conv1_1 = conv1_1.cuda().eval()

        conv2_1 = nn.Sequential(*net_children[2:7])
        self.conv2_1 = conv2_1.cuda().eval()

        conv3_1 = nn.Sequential(*net_children[7:12])
        self.conv3_1 = conv3_1.cuda().eval()

        conv4_1 = nn.Sequential(*net_children[12:21])
        self.conv4_1 = conv4_1.cuda().eval()

        conv5_1 = nn.Sequential(*net_children[21:30])
        self.conv5_1 = conv5_1.cuda().eval()

    def feed_forward(self, im_in):
        out1 = self.conv1_1(im_in)
        out2 = self.conv2_1(out1) 
        out3 = self.conv3_1(out2)
        out4 = self.conv4_1(out3)
        out5 = self.conv5_1(out4)

        return out1, out2, out3, out4, out5
    
    def compute_matrix(self, fea_map):
        shape = fea_map.shape
        fea_map = fea_map.view(shape[0], shape[1], -1).contiguous()
        #zero-mean
        mean = torch.mean(fea_map, dim=-1, keepdim=True)
        mean = mean.repeat(1,1,fea_map.shape[-1])
        fea_map = fea_map - mean
        fea_map_t = fea_map.permute(0,2,1)
        
        matrix = torch.bmm(fea_map, fea_map_t)
        return matrix

    def compute_simi(self, map1, map2):
        bs = map1.shape[0]
        map_vec1 = map1.view(bs, -1).contiguous()
        map_vec2 = map2.view(bs, -1).contiguous()        

        norm1 = torch.norm(map_vec1, dim=1)
        norm2 = torch.norm(map_vec2, dim=1)
    
        return torch.sum(map_vec1 * map_vec2, dim=1) / (norm1 * norm2 + 1e-10)

    def forward(self, left_im, right_im, cond_im):
        r_t = self.feed_forward(right_im)
        l_t = self.feed_forward(left_im)
        c_t = self.feed_forward(cond_im)
        
        bs = l_t[0].shape[0]
        l_ini_mat = self.compute_matrix(l_t[0])
        r_ini_mat = self.compute_matrix(r_t[0])
        c_ini_mat = self.compute_matrix(c_t[0])

        p_sim = self.compute_simi(r_ini_mat, l_ini_mat)
        n_sim = self.compute_simi(r_ini_mat, c_ini_mat)
        n_sim1 = self.compute_simi(l_ini_mat, c_ini_mat)

        loss = torch.nn.functional.relu(n_sim - p_sim + self.margin)
        loss += torch.nn.functional.relu(n_sim1 - p_sim + self.margin) 

        for i in range(1, len(l_t)):
                l_ini_mat = self.compute_matrix(l_t[i])
                r_ini_mat = self.compute_matrix(r_t[i])
                c_ini_mat = self.compute_matrix(c_t[i])

                p_sim = self.compute_simi(r_ini_mat, l_ini_mat)
                n_sim = self.compute_simi(r_ini_mat, c_ini_mat)
                n_sim1 = self.compute_simi(l_ini_mat, c_ini_mat)

                loss += torch.nn.functional.relu(n_sim - p_sim + self.margin) 
                loss += torch.nn.functional.relu(n_sim1 - p_sim + self.margin) 
        
        return torch.mean(loss)
