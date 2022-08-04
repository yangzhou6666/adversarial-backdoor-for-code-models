from collections import OrderedDict
import torch
import torch.nn as nn
from ncc.models.ncc_model import NccEncoderDecoderModel
from ncc.modules.embedding import Embedding
from ncc.modules.code2vec.mm_encoder import MMEncoder
from ncc.modules.seq2seq.mm_decoder import MMLSTMDecoder
from ncc.models import register_model
from ncc.utils import utils
DEFAULT_MAX_SOURCE_POSITIONS = None #1e5
DEFAULT_MAX_TARGET_POSITIONS = None #1e5


@register_model('mm2seq')
class MM2SeqModel(NccEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
    #
    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument('--dropout', type=float, metavar='D',
    #                         help='dropout probability')
    #     parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
    #                         help='encoder embedding dimension')
    #     parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained encoder embedding')
    #     parser.add_argument('--encoder-freeze-embed', action='store_true',
    #                         help='freeze encoder embeddings')
    #     parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
    #                         help='encoder hidden size')
    #     parser.add_argument('--encoder-layers', type=int, metavar='N',
    #                         help='number of encoder layers')
    #     parser.add_argument('--encoder-bidirectional', action='store_true',
    #                         help='make all layers of encoder bidirectional')
    #     parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
    #                         help='decoder embedding dimension')
    #     parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained decoder embedding')
    #     parser.add_argument('--decoder-freeze-embed', action='store_true',
    #                         help='freeze decoder embeddings')
    #     parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
    #                         help='decoder hidden size')
    #     parser.add_argument('--decoder-layers', type=int, metavar='N',
    #                         help='number of decoder layers')
    #     parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
    #                         help='decoder output embedding dimension')
    #     parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
    #                         help='decoder attention')
    #     parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
    #                         help='comma separated list of adaptive softmax cutoff points. '
    #                              'Must be used with adaptive_loss criterion')
    #     parser.add_argument('--share-decoder-input-output-embed', default=False,
    #                         action='store_true',
    #                         help='share decoder input and output embeddings')
    #     parser.add_argument('--share-all-embeddings', default=False, action='store_true',
    #                         help='share encoder, decoder and output embeddings'
    #                              ' (requires shared dictionary and embed dim)')
    #
    #     # Granular dropout settings (if not specified these default to --dropout)
    #     parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
    #                         help='dropout probability for encoder input embedding')
    #     parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
    #                         help='dropout probability for encoder output')
    #     parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
    #                         help='dropout probability for decoder input embedding')
    #     parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
    #                         help='dropout probability for decoder output')
    #     # fmt: on

    def load_pretrained_embedding_from_file(sefl, args, embed_path, dictionary, embed_dim):
        if args['model']['encoder_embed_path']:
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            pretrained_encoder_embed = utils.load_embedding(embed_dict, dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        # base_architecture(args)

        if args['model']['encoder_layers'] != args['model']['decoder_layers']:
            raise ValueError('--encoder-layers must match --decoder-layers')

        # max_source_positions = getattr(args, 'max_source_positions', DEFAULT_MAX_SOURCE_POSITIONS)
        # max_target_positions = getattr(args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)
        max_source_positions = args['model']['max_source_positions'] if args['model']['max_source_positions'] else DEFAULT_MAX_SOURCE_POSITIONS
        max_target_positions = args['model']['max_target_positions'] if args['model']['max_target_positions'] else DEFAULT_MAX_TARGET_POSITIONS

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim, modality='text'):
            if modality == 'code':
                num_embeddings = len(dictionary['code'])
                padding_idx = dictionary['code'].pad()
                embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
                embed_dict = utils.parse_embedding(embed_path)
                utils.print_embed_overlap(embed_dict, dictionary)
                pretrained_embedding = utils.load_embedding(embed_dict, dictionary, embed_tokens)
            if modality == 'path':
                # pretrained embedding for border
                num_embeddings = len(dictionary['path'][0])
                padding_idx = dictionary['path'][0].pad()
                embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
                embed_dict = utils.parse_embedding(embed_path)
                utils.print_embed_overlap(embed_dict, dictionary)
                pretrained_embedding_border = utils.load_embedding(embed_dict, dictionary, embed_tokens)
                # pretrained embedding for center
                num_embeddings = len(dictionary['path'][1])
                padding_idx = dictionary['path'][0].pad()
                embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
                embed_dict = utils.parse_embedding(embed_path)
                utils.print_embed_overlap(embed_dict, dictionary)
                pretrained_embedding_center = utils.load_embedding(embed_dict, dictionary, embed_tokens)
                pretrained_embedding = (pretrained_embedding_border, pretrained_embedding_center)
            if modality == 'ast':
                pass
            return pretrained_embedding

        # 1. laod pretrained_encoder_embed
        pretrained_encoder_embed = {modality: None for modality in args['task']['source_lang']}
        for modality in args['task']['source_lang']:
            if modality == 'path':
                num_embeddings = len(task.source_dictionary['path'][0])  # TODO
                pretrained_encoder_embed_border = Embedding(
                    num_embeddings, args['model']['encoder_embed_dim'], task.source_dictionary['path'][0].pad()
                )
                num_embeddings = len(task.source_dictionary['path'][0])  # TODO
                pretrained_encoder_embed_center = Embedding(
                    num_embeddings, args['model']['encoder_embed_dim'], task.source_dictionary['path'][1].pad()
                )
                pretrained_encoder_embed[modality] = (pretrained_encoder_embed_border, pretrained_encoder_embed_center)
            else:
                num_embeddings = len(task.source_dictionary[modality])  # TODO
                pretrained_encoder_embed[modality] = Embedding(
                    num_embeddings, args['model']['encoder_embed_dim'], task.source_dictionary[modality].pad()
                )
        if args['model']['encoder_embed_path']:
            pretrained_encoder_embed['code'] = load_pretrained_embedding_from_file(
                args['model']['encoder_embed_path'], task.source_dictionary, args['model']['encoder_embed_dim'], modality='code')
        if args['model']['encoder_embed_path_path']:
            pretrained_encoder_embed['path'] = load_pretrained_embedding_from_file(
                args['model']['encoder_embed_path_path'], task.source_dictionary, args['model']['encoder_embed_dim'],
                modality='path')
        if args['model']['encoder_embed_path_ast']:
            pretrained_encoder_embed['path'] = load_pretrained_embedding_from_file(
                args['model']['encoder_embed_path_ast'], task.source_dictionary, args['model']['encoder_embed_dim'],
                modality='ast')
        # else:
        #     num_embeddings = len(task.source_dictionary['path']) # TODO
        #     pretrained_encoder_embed = Embedding(
        #         num_embeddings, args['model']['encoder_embed_dim'], task.source_dictionary['path'].pad()
        #     )

        # 2. share embedding
        if args['model']['share_all_embeddings']:
            # double check all parameters combinations are valid
            if task.source_dictionary['code'] != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args['model']['decoder_embed_path'] and (
                    args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args['model']['share_decoder_input_output_embed'] = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args['model']['decoder_embed_path']:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args['model']['decoder_embed_path'],
                    task.target_dictionary,
                    args['model']['decoder_embed_dim']
                )

        # 3. one last double check of parameter combinations
        if args['model']['share_decoder_input_output_embed'] and (
                args['model']['decoder_embed_dim'] != args['model']['decoder_out_embed_dim']):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        # 4. free word embedding
        if args['model']['encoder_freeze_embed']:
            for modality in args['task']['source_lang']:
                if modality == 'path':
                    pretrained_encoder_embed['path'][0].weight.requires_grad = False
                    pretrained_encoder_embed['path'][1].weight.requires_grad = False
                else:
                    pretrained_encoder_embed[modality].weight.requires_grad = False
        if args['model']['decoder_freeze_embed']:
            pretrained_decoder_embed.weight.requires_grad = False

        # encoder = LSTMEncoder(
        #     dictionary=task.source_dictionary['code'],
        #     embed_dim=args['model']['encoder_embed_dim'],
        #     hidden_size=args['model']['encoder_hidden_size'],
        #     num_layers=args['model']['encoder_layers'],
        #     dropout_in=args['model']['encoder_dropout_in'],
        #     dropout_out=args['model']['encoder_dropout_out'],
        #     bidirectional=bool(args['model']['encoder_bidirectional']),
        #     pretrained_embed=pretrained_encoder_embed,
        #     max_source_positions=max_source_positions
        # )
        encoder = MMEncoder(
            args,
            dictionary=task.source_dictionary,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions
        )
        decoder = MMLSTMDecoder(
            dictionary=task.target_dictionary,
            src_modalities =args['task']['source_lang'],
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

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out