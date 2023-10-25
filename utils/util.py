import torch
from .dataset import (
  create_input_mask,
  create_output_mask,
  create_padding_mask
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(dataloader, model, criterion, optimizer, scheduler):
	model.train()
	losses = 0

	for item in tqdm(dataloader):
		inputs = item[0].to(device)
		target = item[1].to(device)

		target_input = target[:, :-1]

		input_mask = create_input_mask(inputs).to(device)
		output_mask = create_output_mask(target_input).to(device)
		input_padding_mask = create_padding_mask(inputs, 1).to(device)
		output_padding_mask = create_padding_mask(target_input, 1).to(device)

		logits = model(
			inputs, target_input,
			input_mask, output_mask,
			input_padding_mask, output_padding_mask,
			input_padding_mask
		)

		target_output = target[:, 1:]

		optimizer.zero_grad()
		loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
		loss.backward()
		optimizer.step()
		scheduler.step()

		losses += loss.item()

	return losses / len(list(dataloader))


def eval_model(dataloader, model, criterion):
	model.eval()
	losses = 0

	with torch.no_grad():
		for item in tqdm(dataloader):
			inputs = item[0].to(device)
			target = item[1].to(device)

			target_input = target[:, :-1]

			input_mask = create_input_mask(inputs).to(device)
			output_mask = create_output_mask(target_input).to(device)
			input_padding_mask = create_padding_mask(inputs, 1).to(device)
			output_padding_mask = create_padding_mask(target_input, 1).to(device)

			logits = model(
				inputs, target_input,
				input_mask, output_mask,
				input_padding_mask, output_padding_mask,
				input_padding_mask
			)

			target_output = target[:, 1:]

			loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))

			losses += loss.item()

	return losses / len(list(dataloader))
