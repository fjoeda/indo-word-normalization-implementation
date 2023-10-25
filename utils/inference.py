from .model import Seq2SeqTransformer
from .char_encoder import CharacterEncoder
import torch


class ModelInference:
    def __init__(self, model_dir, device, config) -> None:
        self.char_encoder = CharacterEncoder()
        self.model = Seq2SeqTransformer(
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            n_head=config['num_head'],
            emb_dim=config['embedding_dim'],
            feedforward_dim=config['feedforward_dim'],
            input_vocab_size=self.char_encoder.get_total_char(),
            output_vocab_size=self.char_encoder.get_total_char(),
            dropout=config['dropout']
        )
        model_state_dict = torch.load