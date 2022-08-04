# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.common.layers import (
    Embedding,
    Linear,
    LSTM,
)


class SeqEncoder(NccEncoder):
    def __init__(self, dictionary, embed_dim,
                 hidden_dim, rnn_layers=1, bidirectional=True,
                 dropout=0.25):
        super(SeqEncoder, self).__init__(dictionary)
        self.padding_idx = self.dictionary.pad()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)
        self.rnn = LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        bsz = src_tokens.size(0)
        x = self.embed(src_tokens)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        if src_lengths is None:
            src_lengths = src_lengths.new([src_lengths.size(0)]).copy_(
                src_tokens.ne(self.padding_idx).sum(-1)
            )

        # sort
        sorted_lens, indices = src_lengths.sort(descending=True)
        sorted_x = x.index_select(0, indices)
        sorted_x = pack_padded_sequence(sorted_x, sorted_lens.data.tolist(), batch_first=True)

        x, (h, c) = self.rnn(sorted_x)

        _, reversed_indices = indices.sort()
        # x, lens = pad_packed_sequence(x, batch_first=True)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = x.index_select(0, reversed_indices)
        h = h.index_select(1, reversed_indices)
        h = h.view(self.rnn_layers, 2 if self.bidirectional else 1, bsz, self.hidden_dim)
        h = h[-1].view(bsz, -1)
        return h


class NBOWEncoder(NccEncoder):
    def __init__(self, dictionary, embed_dim, dropout=0.25):
        super(NBOWEncoder, self).__init__(dictionary)
        self.padding_idx = self.dictionary.pad()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.embed = Embedding(len(dictionary), embed_dim)  # , padding_idx=self.padding_idx)
        self.init_weights()

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        lens = src_tokens.size(1)
        x = self.embed(src_tokens)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.max_pool1d(x.transpose(1, 2), lens).squeeze(2)
        return x
