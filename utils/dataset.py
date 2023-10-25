from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils import data
import torch


def prepare_dataset(dataset, char_encoder):
  inputs = []
  targets = []

  for item in dataset:
    inp = char_encoder.encode(item['slang'])
    tgt = char_encoder.encode(item['formal'])
    inputs.append(inp)
    targets.append(tgt)

  pad_input = pad_sequence(inputs, batch_first=True, padding_value=1)
  pad_output = pad_sequence(targets, batch_first=True, padding_value=1)
  return data.TensorDataset(pad_input,pad_output)

def create_padding_mask(batched_tensor_input, padding_value):
  return batched_tensor_input == padding_value

def create_output_mask(batched_tensor_input):
  size = batched_tensor_input.size(1)
  mask = torch.triu(torch.ones(size, size)).T.float()
  return mask.masked_fill(mask == 1, 0).masked_fill(mask == 0, float("-inf"))

def create_input_mask(batched_tensor_input):
  size = batched_tensor_input.size(1)
  return torch.zeros((size, size)).type(torch.bool)