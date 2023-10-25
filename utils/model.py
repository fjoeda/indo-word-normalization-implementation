import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
  def __init__(self, emb_size, dropout, max_length = 1000):
    super(PositionalEncoding, self).__init__()
    pos_embedding = torch.zeros(max_length, emb_size)
    pos = torch.arange(0, max_length).reshape(max_length, 1)
    pos_val = torch.exp(torch.log(pos) - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
    pos_embedding[:, 0::2] = torch.sin(pos_val)
    pos_embedding[:, 1::2] = torch.cos(pos_val)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("pos_embedding", pos_embedding)

  def forward(self, token_embedding: torch.Tensor):
    return self.dropout(
        token_embedding + self.pos_embedding[:token_embedding.size(1), :]
    )


class TokenEmbedding(nn.Module):
  def __init__(self, num_tokens, emb_size):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(num_tokens, emb_size)
    self.emb_size = emb_size

  def forward(self, num_tokens):
    return self.embedding(num_tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
  def __init__(
      self,
      num_encoder_layers,
      num_decoder_layers,
      emb_dim,
      n_head,
      input_vocab_size,
      output_vocab_size,
      feedforward_dim,
      dropout=0.1
  ):
    super(Seq2SeqTransformer, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.transformers = nn.Transformer(
        d_model=emb_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        nhead=n_head,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        batch_first=True
    )

    self.generator = nn.Linear(emb_dim, output_vocab_size)
    self.inp_token_embedding = TokenEmbedding(input_vocab_size, emb_dim)
    self.out_token_embedding = TokenEmbedding(output_vocab_size, emb_dim)
    self.positional_encoding = PositionalEncoding(emb_dim, dropout)

  def forward(
      self,
      inputs,
      outputs,
      input_mask,
      output_mask,
      input_padding_mask,
      output_padding_mask,
      memory_key_pad_mask
  ):
    inp_embedding = self.positional_encoding(
        self.inp_token_embedding(inputs)
    )
    out_embedding = self.positional_encoding(
        self.out_token_embedding(outputs)
    )

    logits = self.transformers(
        inp_embedding,
        out_embedding,
        input_mask,
        output_mask,
        None,
        input_padding_mask,
        output_padding_mask,
        memory_key_pad_mask
    )

    return self.generator(logits)

  def encode(self, inputs, input_mask):
    return self.transformers.encoder(
        self.positional_encoding(
            self.inp_token_embedding(inputs)
        ),
        input_mask
    )

  def decode(self, output, memory, output_mask):
    return self.transformers.decoder(
        self.positional_encoding(
            self.out_token_embedding(output)
        ),
        memory,
        output_mask
    )