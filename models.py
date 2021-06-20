import torch.nn as nn
import torch
from torchvision.models import resnet152, inception_v3
import torch.nn.functional as F
import pdb
from torchvision.models.resnet import model_urls
from torchvision.models.inception import model_urls as model_urls1
from network import Conditional_Network
from network import ResBlock, Fuse_Whole

model_urls['resnet152'] = model_urls['resnet152'].replace('https://', 'http://')
model_urls1['inception_v3_google'] = model_urls1['inception_v3_google'].replace('https://', 'http://')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
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
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ELU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ELU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ELU(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ELU(),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        #decoding module
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer13 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        )
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        self.posi_emb = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )
        #encode the conditional information
        self.c_net = Conditional_Network() 
        #we need to define the network to fuse the conditionl information
        self.right_fuse1 = ResBlock(128*3, [64, 64, 128])
        self.whole_fuse1 = Fuse_Whole(128, [64,64, 128])
        #fuse the new feature is also necessary
        self.right_fuse2 = ResBlock(128*3, [64, 64, 128])
        self.whole_fuse2 = Fuse_Whole(128, [64, 64, 128])
        
        self.right_fuse3 = ResBlock(64*3, [32, 32, 64])
        self.whole_fuse3 = Fuse_Whole(64, [32, 32, 64])
 
        self.right_fuse4 = ResBlock(64*3, [32, 32, 64])
        self.whole_fuse4 = Fuse_Whole(64, [32, 32, 64])
 
        self.right_fuse5 = ResBlock(32*3, [16, 16, 32])
        self.whole_fuse5 = Fuse_Whole(32, [16, 16, 32])

        #map the map to 3 x 3
        self.prog_map1 = nn.Sequential(
        nn.Conv2d(32*2, 64, kernel_size=1, padding=1, stride=1),
        nn.ELU(),
        nn.Conv2d(32*2, 128, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.AdaptiveAvgPool2d(output_size=(3,3))
        )

        self.prog_map2 = nn.Sequential(
        nn.Conv2d(64*2, 128, kernel_size=1, padding=1, stride=1),
        nn.ELU(),
        nn.Conv2d(64*2, 256, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(256, 64, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d(output_size=(3,3))
        )
        self.prog_map3 = nn.Sequential(
        nn.Conv2d(64*2, 128, kernel_size=1, padding=1, stride=1),
        nn.ELU(),
        nn.Conv2d(64*2, 256, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(256, 64, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d(output_size=(3,3))
        )
        self.prog_map4 = nn.Sequential(
        nn.Conv2d(128*2, 256, kernel_size=1, padding=1, stride=1),
        nn.ELU(),
        nn.Conv2d(128*2, 512, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(512, 128, kernel_size=3, padding=1, stride=2),
        nn.AdaptiveAvgPool2d(output_size=(3,3))
        )
        self.prog_map5 = nn.Sequential(
        nn.Conv2d(128*2, 256, kernel_size=1, padding=1, stride=1),
        nn.ELU(),
        nn.Conv2d(128*2, 512, kernel_size=3, padding=1, stride=2),
        nn.ELU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(512, 128, kernel_size=3, padding=1, stride=2),
        nn.AdaptiveAvgPool2d(output_size=(3,3))
        )

        self.gate = torch.nn.Sigmoid()

    def gate_OP(self, c, l):
        return c * self.gate(l)

    def image_guide_conv(self, rht_pdt, raw_img, ref_img, map_module):
        #raw_img: bs x c_num x h x w
        #ref_img: bs x c_num x h x w
        bs = raw_img.shape
        gate_map = map_module(torch.cat([raw_img, ref_img],dim=1))#bs x c_num x k x k
        #gate_map = self.gate(gate_map)
        #gate_map = torch.nn.softmax(gate_map, dim=1)
        gated_fea_maps = torch.zeros(bs[0], bs[1], bs[2], bs[3]).cuda()
        for i in range(bs[0]):
            temp_ref = ref_img[i].unsqueeze(0)
            temp_map = torch.softmax(gate_map[i], dim=0)
            temp_map = gate_map[i].unsqueeze(1).repeat(1,bs[1],1,1)
            gated_fea_maps[i] = F.conv2d(temp_ref, temp_map, padding=1).squeeze()

        #return raw_img + gated_fea_maps
        gate_lft = self.gate_OP(raw_img, rht_pdt)
        return rht_pdt + gated_fea_maps + gate_lft

    def forward(self, x, sketch, posi, detail_ims):
        lft = x[:,:,:,:128]
        lft = lft[:,:3, :, :]
        x = torch.cat([x, sketch], dim=1)
        out1 = self.layer1(x)
        posi_fea = self.posi_emb(posi)
        out_temp = torch.cat([out1,posi_fea], dim=1)
        out2 = self.layer2(out_temp)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.layer6(out5)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        #encode the condition
        c_out1,s_out1,p_out1, c_out2,s_out2,p_out2, c_out3, s_out3, p_out3, \
        c_out4,s_out4,p_out4, c_out5,s_out5,p_out5 = self.c_net(detail_ims, sketch[:,:,:,128:], posi[:,:,:,128:])
        #get the left operation
        #l_t1, _, _, l_t2, _,_, l_t3, _,_, l_t4,_,_,l_t5,_,_=self.c_net(lft, sketch[:,:,:,:128], posi[:,:,:,:128])
        #fuse the right half--------------
        rht_out = self.image_guide_conv(out[:,:,:,out.shape[-1]//2:],torch.flip(out[:,:,:,:out.shape[-1]//2], [3]), c_out5, self.prog_map5)
        right_half = torch.cat([rht_out, s_out5, p_out5], dim=1)
        right_out = self.right_fuse1(right_half)    
        whole_out = torch.cat([out[:,:,:,:out.shape[-1]//2], right_out], dim=3)        
        out = self.whole_fuse1(whole_out) 
        out = torch.add(out, out5)
        #----------------------------------
        out = self.layer12(out)
        rht_out = self.image_guide_conv(out[:,:,:,out.shape[-1]//2:], torch.flip(out[:,:,:,:out.shape[-1]//2], [3]), c_out4, self.prog_map4)
        right_half = torch.cat([rht_out, s_out4, p_out4], dim=1)
        right_out = self.right_fuse2(right_half)        
        whole_out = torch.cat([out[:,:,:,:out.shape[-1]//2], right_out], dim=3)
        out = self.whole_fuse2(whole_out)
        out = torch.add(out, out4)
        #----------------------------------------
        out = self.layer13(out)
        out = self.layer14(out)
       
        rht_out = self.image_guide_conv(out[:,:,:,out.shape[-1]//2:], torch.flip(out[:,:,:,:out.shape[-1]//2],[3]), c_out3, self.prog_map3)
        right_half = torch.cat([rht_out, s_out3, p_out3], dim=1)
        right_out = self.right_fuse3(right_half)        
        whole_out = torch.cat([out[:,:,:,:out.shape[-1]//2], right_out], dim=3)
        out = self.whole_fuse3(whole_out)
          
        out = torch.add(out, out3)
        #----------------------------------------
        out = self.layer15(out)
        rht_out = self.image_guide_conv(out[:,:,:,out.shape[-1]//2:], torch.flip(out[:,:,:,:out.shape[-1]//2],[3]), c_out2, self.prog_map2)
        right_half = torch.cat([rht_out, s_out2, p_out2], dim=1)
        right_out = self.right_fuse4(right_half)        
        whole_out = torch.cat([out[:,:,:,:out.shape[-1]//2], right_out], dim=3)
        out = self.whole_fuse4(whole_out)

        out = torch.add(out, out2)
        #----------------------------------------
        out = self.layer16(out)
        out = self.layer17(out)

        rht_out = self.image_guide_conv(out[:,:,:,out.shape[-1]//2:], torch.flip(out[:,:,:,:out.shape[-1]//2],[3]), c_out1, self.prog_map1)
        right_half = torch.cat([rht_out, s_out1, p_out1], dim=1)

        right_out = self.right_fuse5(right_half)        
        whole_out = torch.cat([out[:,:,:,:out.shape[-1]//2], right_out], dim=3)
        out = self.whole_fuse5(whole_out)
        out = torch.add(out, out1)
        
        out = self.layer18(out)
        out = self.layer19(out)

        return out

class Discriminator256(nn.Module):
    def __init__(self):
        super(Discriminator256, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(5, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=(2,4), stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        # self.layer9 = nn.Linear(4096, 256, bias=False)
        self.layer10 = nn.Linear(1000, 256, bias=False)

        self.layer11 = nn.Linear(256, 1, bias=False)

    def forward(self, x, y, z, sketch):
        out = torch.cat([x, y, sketch], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
 
        out = self.layer8(out)
        out_t = self.layer11(out)
        z1 = self.layer10(z)
        out = (out * z1).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out

class InceptionExtractor256(nn.Module):
    def __init__(self):
        super(InceptionExtractor256, self).__init__()
        self.inception_v3 = resnet152(pretrained=True)

    def forward(self, x):
        x = self.inception_v3(x)
        return x


class Discriminator512(nn.Module):
    def __init__(self):
        super(Discriminator512, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 256, bias=False)  # It might not be correct
        self.layer10 = nn.Linear(1000, 256, bias=False)
        self.layer11 = nn.Linear(256, 1, bias=False)

    def forward(self, x, y, z):
        out = torch.cat([x, y], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out_t = self.layer11(out)

        z1 = self.layer10(z)
        out = (out * z1).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out

class InceptionExtractor512(nn.Module):
    def __init__(self):
        super(InceptionExtractor512, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True)

    def forward(self, x):
        x = self.inception_v3(x)
        return x
