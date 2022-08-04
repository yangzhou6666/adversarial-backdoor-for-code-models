import torch.nn as nn
import torch.nn.functional as F

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.embedding import Embedding
from ncc.utils import utils
from ncc.modules.code2vec.lstm_encoder import LSTMEncoder
from ncc.modules.code2vec.path_encoder import PathEncoder

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class MMEncoder(NccEncoder):
    """LSTM encoder."""
    def __init__(
        # self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        # dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        # left_pad=True, pretrained_embed=None, padding_idx=None,
        # max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
        self,
        args,
        dictionary,
        pretrained_embed=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super().__init__(dictionary)
        # self.num_layers = num_layers
        # self.dropout_in = dropout_in
        # self.dropout_out = dropout_out
        # self.bidirectional = bidirectional
        # self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        # num_embeddings = len(dictionary)
        # self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        # if pretrained_embed is None:
        #     self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        # else:
        #     self.embed_tokens = pretrained_embed
        self.args = args
        self.dictionary = dictionary

        if 'code' in self.args['task']['source_lang']:
            self.code_encoder = LSTMEncoder(
                dictionary=dictionary['code'],
                embed_dim=args['model']['encoder_embed_dim'],
                hidden_size=args['model']['encoder_hidden_size'],
                num_layers=args['model']['encoder_layers'],
                dropout_in=args['model']['encoder_dropout_in'],
                dropout_out=args['model']['encoder_dropout_out'],
                bidirectional=bool(args['model']['encoder_bidirectional']),
                pretrained_embed=pretrained_embed['code'],
                max_source_positions=max_source_positions
            )

        if 'path' in self.args['task']['source_lang']:
            self.path_encoder = PathEncoder(
                dictionary=dictionary['path'],
                embed_dim=args['model']['encoder_embed_dim'],
                hidden_size=args['model']['encoder_hidden_size'],
                num_layers=args['model']['encoder_layers'],
                dropout_in=args['model']['encoder_dropout_in'],
                dropout_out=args['model']['encoder_dropout_out'],
                bidirectional=bool(args['model']['encoder_bidirectional']),
                pretrained_embed=pretrained_embed['path'],
                max_source_positions=max_source_positions
            )

        if 'bin_ast' in self.args['task']['source_lang']:
            pass

        self.output_units = args['model']['encoder_hidden_size']
        if bool(args['model']['encoder_bidirectional']):
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        code_encoder_out, ast_encoder_out, path_encoder_out = None, None, None
        if 'code' in self.args['task']['source_lang']:
            code_encoder_out = self.code_encoder(src_tokens['code'], src_lengths=src_lengths['code'])

        if 'ast' in self.args['task']['source_lang']:
            # ast_encoder_out = self.ast_encoder()
            pass

        if 'path' in self.args['task']['source_lang']:
            path_encoder_out = self.path_encoder(src_tokens['path'], src_lengths=src_lengths['path'])

        # if self.config['training']['enc_hc2dec_hc'] is None:
        #     dec_hc = None
        # else:
        #     dec_hc = self._enc_hc2dec_hc(enc_hidden_state)
        # return enc_output, dec_hc, enc_mask

        return {
            'code': code_encoder_out,
            'ast': ast_encoder_out,
            'path': path_encoder_out
        }
        # {
        #     'encoder_out': (x, final_hiddens, final_cells),
        #     'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        # }


    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions