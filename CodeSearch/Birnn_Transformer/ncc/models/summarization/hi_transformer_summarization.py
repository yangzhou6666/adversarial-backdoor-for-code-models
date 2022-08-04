# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from fairseq import utils
# from fairseq.modules import (
#     LearnedPositionalEmbedding, MultiheadAttention,
#     SinusoidalPositionalEmbedding,
# )
from ncc.modules.roberta.learned_positional_embedding import LearnedPositionalEmbedding
from ncc.modules.attention.multihead_attention import MultiheadAttention
from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ncc.modules.seq2seq.ncc_incremental_decoder import NccIncrementalDecoder
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.models.ncc_model import NccEncoderDecoderModel
from ncc.models import register_model
from ncc.utils import utils


# from . import (
#     NccIncrementalDecoder, NccEncoder, NccModel,
#     register_model, register_model_architecture,
# )


# @register_model('hi_transformer_summarization')
# class HiTransformerSummarizationModel(NccModel):


def get_sent_end_repr(src_emb, sent_ends):
    bsz, nsent = sent_ends.size()
    assert bsz == src_emb.size(0)
    seqlen = src_emb.size(1)
    offset = torch.linspace(0, (bsz - 1) * seqlen, bsz).type(sent_ends.type())
    sent_ends_abs = sent_ends + offset.view(-1, 1)
    sent_ends_repr = src_emb.contiguous().view(bsz * seqlen, -1)[sent_ends_abs]
    sent_ends_repr = sent_ends_repr.view(bsz, nsent, -1)

    return sent_ends_repr


@register_model('hi_transformer_summarization')
class HiTransformerSummarizationModel(NccEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     parser.add_argument('--dropout', type=float, metavar='D',
    #                         help='dropout probability')
    #     parser.add_argument('--attention-dropout', type=float, metavar='D',
    #                         help='dropout probability for attention weights')
    #     parser.add_argument('--relu-dropout', type=float, metavar='D',
    #                         help='dropout probability after ReLU in FFN')
    #     parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained encoder embedding')
    #     parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
    #                         help='encoder embedding dimension')
    #     parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
    #                         help='encoder embedding dimension for FFN')
    #     parser.add_argument('--encoder-layers', type=int, metavar='N',
    #                         help='num encoder layers')
    #     parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
    #                         help='num encoder attention heads')
    #     parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
    #                         help='apply layernorm before each encoder block')
    #     parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
    #                         help='use learned positional embeddings in the encoder')
    #     parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained decoder embedding')
    #     parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
    #                         help='decoder embedding dimension')
    #     parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
    #                         help='decoder embedding dimension for FFN')
    #     parser.add_argument('--decoder-layers', type=int, metavar='N',
    #                         help='num decoder layers')
    #     parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
    #                         help='num decoder attention heads')
    #     parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
    #                         help='use learned positional embeddings in the decoder')
    #     parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
    #                         help='apply layernorm before each decoder block')
    #     parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
    #                         help='share decoder input and output embeddings')
    #     parser.add_argument('--share-all-embeddings', default=False, action='store_true',
    #                         help='share encoder, decoder and output embeddings'
    #                              ' (requires shared dictionary and embed dim)')

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        # base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args['model']['share_all_embeddings']:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args['model']['decoder_embed_path'] and (
                args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
            )
            decoder_embed_tokens = encoder_embed_tokens
            args['model']['share_decoder_input_output_embed'] = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args['model']['decoder_embed_dim'], args['model']['decoder_embed_path']
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return HiTransformerSummarizationModel(encoder, decoder)

    # def forward(self, src_tokens, doc_pad_mask, doc_pos_tok, masked_sent_positions, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, doc_pad_mask, doc_pos_tok)
    #     decoder_out = self.decoder(encoder_out, masked_sent_positions, prev_output_tokens)
    #     return decoder_out
    def forward(self, src_tokens, src_sent_ends, doc_pad_mask, doc_pos_tok, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_sent_ends, doc_pad_mask, doc_pos_tok)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class TransformerEncoder(NccEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args['model']['dropout']

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args['model']['encoder_learned_pos'],
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args['model']['encoder_layers'])
        ])

        self.sent_embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=False,
            learned=args['model']['encoder_learned_pos'],
        )

        self.doc_layers = nn.ModuleList([])
        self.doc_layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args['model']['encoder_layers'])
        ])

    def forward(self, src_tokens, src_sent_ends, doc_pad_mask, doc_pos_tok):
        bsz, seqlen = src_tokens.size()
        # src_tokens = src_tokens.view(bsz * n_sent, seqlen)
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        doc_pos = self.sent_embed_positions(doc_pos_tok)

        # sent_repr = x[-1].view(bsz, n_sent, -1)
        x = x.transpose(0, 1)
        sent_repr = get_sent_end_repr(x, src_sent_ends)
        # print('sent_repr', sent_repr.size())
        sent_repr = sent_repr + doc_pos
        # print('sent_repr after', sent_repr.size())
        # n_sent x bsz x C
        sent_repr = sent_repr.transpose(0, 1)
        for doc_layer in self.doc_layers:
            sent_repr = doc_layer(sent_repr, doc_pad_mask)

        return {
            'encoder_out': sent_repr,  # n_sent x bsz x C
            'encoder_padding_mask': doc_pad_mask,  # bsz x n_sent
        }

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_out'] is not None:
            encoder_out_dict['encoder_out'] = \
                encoder_out_dict['encoder_out'].index_select(1, new_order)
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            if 'encoder.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerDecoder(NccIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args['model']['dropout']
        self.share_input_output_embed = args['model']['share_decoder_input_output_embed']

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args['model']['decoder_learned_pos'],
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(args['model']['decoder_layers'])
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        return x, attn

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerDecoder_(NccIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.dropout = args['model']['dropout']

        embed_dim = args['model']['decoder_embed_dim']
        self.out_proj = Linear(embed_dim, len(dictionary))

    def forward(self, encoder_out, incremental_state=None):
        x = encoder_out['encoder_out']

        x = F.dropout(x, p=self.dropout, training=self.training)
        # embed positions
        x = self.out_proj(x)
        x = x.transpose(0, 1)

        return x, x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return 1024

    def upgrade_state_dict(self, state_dict):
        '''
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        '''
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args['model']['encoder_embed_dim']
        # self.self_attn = MultiheadAttention(
        #     self.embed_dim, args['model']['encoder_attention_heads'],
        #     dropout=args['model']['attention_dropout'],
        # )
        # TODO: to be verified
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args['model']['encoder_attention_heads'],
            dropout=args['model']['attention_dropout'],
            # add_bias_kv=add_bias_kv,
            # add_zero_attn=add_zero_attn,
            self_attention=True
        )
        self.dropout = args['model']['dropout']
        self.relu_dropout = args['model']['relu_dropout']
        self.normalize_before = args['model']['encoder_normalize_before']
        self.fc1 = Linear(self.embed_dim, args['model']['encoder_ffn_embed_dim'])
        self.fc2 = Linear(args['model']['encoder_ffn_embed_dim'], self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args['model']['decoder_embed_dim']
        self.self_attn = MultiheadAttention(
            self.embed_dim, args['model']['decoder_attention_heads'],
            dropout=args['model']['attention_dropout'],
        )
        self.dropout = args['model']['dropout']
        self.relu_dropout = args['model']['relu_dropout']
        self.normalize_before = args['model']['decoder_normalize_before']
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args['model']['decoder_attention_heads'],
            dropout=args['model']['attention_dropout'],
        )
        self.fc1 = Linear(self.embed_dim, args['model']['decoder_ffn_embed_dim'])
        self.fc2 = Linear(args['model']['decoder_ffn_embed_dim'], self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            # mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m
