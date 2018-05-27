import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from math import sqrt
from torch.autograd import Variable


class DRRN(nn.Module):

    def __init__(self,opt):
        super(DRRN, self).__init__()
        self.opt = opt
        self.block_num = opt.BlockNum
        self.block_size = opt.BlockSize
        self.shuflle_size = 2

        print ("===>block size",self.block_size)
        print("===>block size", self.block_num)

        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        # (for DRRN)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        # 3 x 3 kernel conv(for dense net)
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        # 1 x 1 kernel conv
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_cat_local = nn.Conv2d(in_channels=128*(self.block_size+1), out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_cat_global = nn.Conv2d(in_channels=128*(self.block_num+1), out_channels=128, kernel_size=1,stride=1, padding=0, bias=False)

        #in original DENSE NET , there are 3 channels in output , but here we use 1 channel to fit our code ?Consider: Why we use only 1 channel?
        self.output_3 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bn = torch.nn.BatchNorm2d(128).cuda()
        #ESPCNN
        self.shuffle = torch.nn.PixelShuffle(self.shuflle_size)
        in_channels = 128
        out_channels = in_channels*(self.shuflle_size * self.shuflle_size)
        self.espcn_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.AvgPool = nn.AvgPool2d(kernel_size=2)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        def ESPCN(x):
            ret = self.espcn_conv(x)
            ret = self.shuffle(ret)
            ret = self.AvgPool(ret)
            ret = self.espcn_conv(ret)
            ret = self.shuffle(ret)
            ret = self.AvgPool(ret)
            return ret

        if self.opt.NetStructure == "DRRN":
            # DRRN
            residual = x
            inputs = self.input(self.relu(x))
            out = inputs
            for _ in range(self.opt.DRRNsize):
                if self.opt.batchNormalize:
                    relu_1 = self.relu(self.conv_bn(out))
                    relu_2 = self.relu(self.conv_bn(self.conv1(relu_1)))
                else:
                    relu_1 = self.relu(out)
                    relu_2 = self.relu(self.conv1(relu_1))
                out = self.conv2(relu_2)
                out = torch.add(out, inputs)




            if self.opt.ESPCN :
                out = ESPCN(out)

            out = self.output(self.relu(out))
            out = torch.add(out, residual)
            return out

        else:
            # DENSE NET
            block_size = self.block_size
            block_num = self.block_num

            def residual_block(x):
                out_seq=[]
                input = x

                out_seq.append(input)

                for i in range(block_size):
                    if self.opt.batchNormalize:
                        l1=self.conv(input)
                        out = self.relu(l1)
                        del (l1)
                    else:
                        l1 = self.conv(input)
                        l2 = self.conv_bn(l1)
                        del(l1)
                        out = self.relu(l2)
                        del(l2)
                    out_seq.append(out)
                    input = torch.add(out,input)

                    del (out)
                cat = torch.cat(out_seq,1)
                out = self.relu(self.conv_cat_local(cat))
                out = torch.add(out,x)

                del (out_seq)  # release memory
                del (cat)
                del (input)

                return out

            F_1 = self.relu(self.input(x))
            F0 = self.relu(self.conv(F_1))
            input = F0
            RDBs = []
            for i in range(block_num):
                out = residual_block(input)
                del (input)
                RDBs.append(out)
                input = out
            RDBs.append(out)# output of the last RDB
            cat = torch.cat(RDBs,1)
            F_GF = self.relu(self.conv(self.relu(self.conv_cat_global(cat))))# 1 x 1 and 3 x 3 conv
            F_DF = torch.add(F_GF,F_1)

            if self.opt.ESPCN:
                F_DF = ESPCN(F_DF)
            ret = self.output(F_DF)


            del (RDBs)  # release memory
            del (input)
            del (out)
            del (F0)
            del (F_1)
            del (cat)
            del (F_GF)
            del (F_DF)

            return ret
