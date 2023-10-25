from .model import Seq2SeqTransformer
from .char_encoder import CharacterEncoder
import torch
from .dataset import create_output_mask

char_encoder = CharacterEncoder()

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
        self.device = device
        self.model.to(device)
        model_state_dict = torch.load(model_dir, map_location=device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def greedy_decode(self, inputs, input_mask, max_length):
        inputs = inputs.to(self.device)
        input_mask = input_mask.to(self.device)
        memory = self.model.encode(inputs, input_mask)
        ys = torch.ones(1, 1).fill_(char_encoder.char2idx["<sos>"]).type(torch.long).to(self.device)
        for i in range(max_length - 1):
            memory = memory.to(self.device)
            output_mask = (create_output_mask(ys).type(torch.bool)).to(self.device)

            out = self.model.decode(ys, memory, output_mask)
            prob = self.model.generator(out[:, -1])

            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([
                ys,
                torch.ones(1, 1).type_as(inputs.data).fill_(next_word.item())
            ], dim=1)

            if next_word.item() == char_encoder.char2idx["<eos>"]:
                break

        return ys

    def normalize(self, word):
        inputs = char_encoder.encode(word).unsqueeze(0)
        num_tokens = inputs.shape[1]
        input_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        out_tokens = self.greedy_decode(
            inputs,
            input_mask,
            max_length=num_tokens + 10,
        )

        norm_tokens = [char_encoder.idx2char[item.item()] for item in out_tokens[0]]

        return "".join(norm_tokens).replace("<sos>","").replace("<eos>","")
