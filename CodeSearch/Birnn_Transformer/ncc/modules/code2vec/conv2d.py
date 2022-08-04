# -*- coding: utf-8 -*-
from ncc.modules.code2vec.base import Encoder_Emb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class Conv2dEncoder(nn.Module):
    '''
    CodeSearchNet baseline:
    conv1d -> activation func -> dropout
    '''
    def __init__(self, token_num: int, in_channels: int, out_channels: int, kernels: List, ) -> None:
        super(Conv2dEncoder, self).__init__()
        # conv1d/conv2d params
        # in text encoder, in_channels=embed_size
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        #
        # # padding
        # assert padding in ['valid', 'same', ], \
        #     LOGGER.error('only [valid/same] padding methods are available, but got {}'.format(padding))
        # self.padding = padding

        # to set the size of output after conv1d be same as the size of input before conv1d
        # self.conv1d = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, )
        self.in_channels = in_channels # embed_size # 300
        self.out_channels = out_channels # kernel_num # 512
        self.kernels = kernels # [2, 3, 4, 5]
        # self.convs = [nn.Conv2d(1, self.kernel_num, (k, self.embed_size)) for k in self.kernels]
        self.wemb = Encoder_Emb(token_num, in_channels)

        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channels, (k, self.in_channels)) for k in self.kernels])


    @classmethod
    def load_from_config(cls, args: Dict) -> Any:
        instance = cls(
            token_num=args['training']['token_num']['comment'],
            in_channels=args['training']['embed_size'],
            out_channels=args['training']['conv2d_out_channels'],
            kernels=args['training']['conv2d_kernels'],
        )
        return instance


    def forward(self, input: torch.Tensor, ) -> Any:
        # (batch, seq_len, emb_size) -> (batch, 1, seq_len, emb_size)
        input_emb = self.wemb(input)
        print('input_emb: ', input_emb.type(), input_emb.size())
        print('input_emb')
        print('self.convs: ', self.convs)
        # assert False
        input_emb = input_emb.unsqueeze(1)
        l= []
        for conv in self.convs:
            em = F.relu(conv(input_emb))#.squeeze(3)
            emb = em.view(em.size(0), em.size(1), em.size(2))
            print('emb: ', emb.size())
            ee = F.max_pool1d(emb, emb.size(2))#.squeeze(2)
            print('ee: ', ee.size())
            # eeee = ee.squeeze(2)
            eeee = ee.view(ee.size(0), ee.size(1))
            print('eeee: ', eeee.size())
            l.append(eeee)
        print('l: ')
        # l = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in l]

        input_emb = torch.cat(l, 1)
        # print(l)
        # input_emb = [F.relu(conv(input_emb)).squeeze(3) for conv in self.convs]
        # input_emb = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in input_emb]
        # input_emb = torch.cat(input_emb, 1)
        # print('input_emb-: ')
        # print(input_emb)

        return input_emb


# if __name__ == '__main__':
#     input = torch.LongTensor([[1, 2, 4, 0], [4, 3, 0, 0]])
#     input_mask = input.data.gt(0).float()
#     embed = nn.Embedding(10, 10, 0)
#     input = embed(input)
#     print('input: ', input.size())
#     encoder = Encoder_Conv2d(in_channels=10, out_channels=20, kernels=[1,2,3], )
#     print(input.size())
#     input = encoder(input)
#     print(input.size())
#
#     input.mean().backward()
