import torch
import torch.nn as nn
from ncc.utils import utils
from ncc.models import register_model
from ncc.models.ncc_model import NccEncoderDecoderModel
from ncc.modules.embedding import Embedding
from ncc.modules.code2vec.path_encoder import PathEncoder
from ncc.modules.seq2seq.path_decoder import LSTMDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('code2seq')
class Code2Seq(NccEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
        if args['model']['encoder_layers'] != args['model']['decoder_layers']:
            raise ValueError('--encoder-layers must match --decoder-layers')

        max_source_positions = args['model']['max_source_positions'] if args['model']['max_source_positions'] \
            else DEFAULT_MAX_SOURCE_POSITIONS
        max_target_positions = args['model']['max_target_positions'] if args['model']['max_target_positions'] \
            else DEFAULT_MAX_TARGET_POSITIONS

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        # path
        if args['model']['encoder_path_embed']:
            pretrained_encoder_path_embed = load_pretrained_embedding_from_file(
                args['model']['encoder_path_embed'],
                task.source_dictionary['path'], args['model']['encoder_path_embed_dim'])
        else:
            num_embeddings = len(task.source_dictionary['path'])
            pretrained_encoder_path_embed = Embedding(
                num_embeddings, args['model']['encoder_path_embed_dim'],
                padding_idx=task.source_dictionary['path'].pad()
            )
        # path.terminals
        if args['model']['encoder_terminals_embed']:
            pretrained_encoder_terminals_embed = load_pretrained_embedding_from_file(
                args['model']['encoder_terminals_embed'],
                task.source_dictionary['path.terminals'], args['model']['encoder_terminals_embed_dim'])
        else:
            num_embeddings = len(task.source_dictionary['path.terminals'])
            pretrained_encoder_terminals_embed = Embedding(
                num_embeddings, args['model']['encoder_terminals_embed_dim'],
                padding_idx=task.source_dictionary['path.terminals'].pad()
            )
        # decoder
        if args['model']['decoder_embed']:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args['model']['decoder_embed'],
                task.target_dictionary, args['model']['decoder_embed_dim'])
        else:
            num_embeddings = len(task.target_dictionary)
            pretrained_decoder_embed = Embedding(
                num_embeddings, args['model']['decoder_embed_dim'],
                padding_idx=task.target_dictionary.pad()
            )

        if args['model']['encoder_path_freeze_embed']:
            pretrained_encoder_path_embed.weight.requires_grad = False
        if args['model']['encoder_terminals_freeze_embed']:
            pretrained_encoder_terminals_embed.weight.requires_grad = False
        if args['model']['decoder_freeze_embed']:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = PathEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args['model']['encoder_path_embed_dim'],
            t_embed_dim=args['model']['encoder_terminals_embed_dim'],
            hidden_size=args['model']['encoder_hidden_size'],
            decoder_hidden_size=args['model']['decoder_hidden_size'],
            num_layers=args['model']['encoder_layers'],
            dropout_in=args['model']['encoder_dropout_in'],
            dropout_out=args['model']['encoder_dropout_out'],
            bidirectional=bool(args['model']['encoder_bidirectional']),
            pretrained_path_embed=pretrained_encoder_path_embed,
            pretrained_terminals_embed=pretrained_encoder_terminals_embed,
            max_source_positions=max_source_positions
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args['model']['decoder_embed_dim'],
            hidden_size=args['model']['decoder_hidden_size'],
            out_embed_dim=args['model']['decoder_out_embed_dim'],
            num_layers=args['model']['decoder_layers'],
            dropout_in=args['model']['decoder_dropout_in'],
            dropout_out=args['model']['decoder_dropout_out'],
            attention=args['model']['decoder_attention'],
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args['model']['share_decoder_input_output_embed'],
            adaptive_softmax_cutoff=(
                args['model']['adaptive_softmax_cutoff']
                if args['criterion'] == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions
        )
        return cls(encoder, decoder)
