# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ncc import LOGGER
from ncc.modules.code2vec.base.util import *
from typing import Dict, Any


class Conv1dEncoder(nn.Module):
    '''
    CodeSearchNet baseline:
    conv1d -> activation func -> dropout
    '''

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str, ) -> None:
        super(Conv1dEncoder, self).__init__()
        # conv1d/conv2d params
        # in text encoder, in_channels=embed_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # padding
        assert padding in ['valid', 'same', ], \
            LOGGER.error('only [valid/same] padding methods are available, but got {}'.format(padding))
        self.padding = padding

        # to set the size of output after conv1d be same as the size of input before conv1d
        self.conv1d = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, )

    @classmethod
    def load_from_config(cls, config: Dict) -> Any:
        instance = cls(
            in_channels=config['training']['embed_size'],
            out_channels=config['training']['out_channels'],
            kernel_size=config['training']['kernel_size'],
        )
        return instance

    def _pad_input(self, input: torch.Tensor, TRG_LEN: int, ) -> torch.Tensor:
        if self.padding == 'valid':
            # valid: feature_size -(conv1d)->  feature_size - (kernel_size-1), if stride=1
            pass
        else:
            # same: try pad same number of 0 at both sides. if fails, pad one more at right
            # ref: tensorflow conv padding mode
            if input.size(-1) - (self.kernel_size - 1) == TRG_LEN:
                pass
            else:
                pad_zero_num = self.kernel_size - 1
                if pad_zero_num % 2 == 0:
                    input = pad_conv1d(input, pad_zero_num // 2, pad_zero_num // 2)
                else:
                    input = pad_conv1d(input, pad_zero_num // 2, pad_zero_num // 2 + 1)
        return input

    def forward_without_transpose(self, input_emb: torch.Tensor, trg_len: int, ) -> Any:
        padded_input_emb = self._pad_input(input_emb, trg_len)
        padded_input_emb = padded_input_emb.contiguous()
        input_emb = self.conv1d(padded_input_emb)
        return input_emb

    def forward(self, input_emb: torch.Tensor, input_mask: torch.Tensor, ) -> Any:
        # (batch, seq_len, emb_size) -> (batch, emb_size, seq_len)
        trg_len = input_emb.size(1)
        input_emb = input_emb.transpose(dim0=1, dim1=2)
        input_emb = self.forward_without_transpose(input_emb, trg_len)
        input_emb = input_emb.transpose(dim0=1, dim1=2)
        input_emb *= input_mask.unsqueeze(dim=-1)
        return input_emb


# if __name__ == '__main__':
#     input = torch.LongTensor([[1, 2, 4, 0], [4, 3, 0, 0]])
#     input_mask = input.data.gt(0).float()
#     embed = nn.Embedding(10, 10, 0)
#     input = embed(input)
#
#     encoder = Encoder_Conv1d(in_channels=10, out_channels=20, kernel_size=2, padding='same', )
#     print(input.size())
#     input = encoder(input, input_mask)
#     print(input.size())
#     input.mean().backward()
