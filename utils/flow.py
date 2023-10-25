import json
from torch.utils import data
from .dataset import (
    prepare_dataset,
    create_padding_mask,
    create_input_mask,
    create_output_mask
)

from .char_encoder import CharacterEncoder
from .model import Seq2SeqTransformer
from .util import train_model, eval_model
from torch import nn, optim
import transformers
import torch

char_encoder = CharacterEncoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_train_dataset(name_sets):
    if name_sets == "indocollex":
        train_ds = json.load(open("./dataset/indo_collex_train.json", "r"))
        val_ds = json.load(open("./dataset/indo_collex_val.json", "r"))
        test_ds = json.load(open("./dataset/indo_collex_test.json", "r"))
    elif name_sets == "col_id_norm":
        train_ds = json.load(open("./dataset/col_id_norm_train.json", "r"))
        val_ds = json.load(open("./dataset/col_id_norm_val.json", "r"))
        test_ds = json.load(open("./dataset/col_id_norm_test.json", "r"))
    elif name_sets == "combined":
        train_ds = json.load(open("./dataset/combined_train.json", "r"))
        val_ds = json.load(open("./dataset/combined_val.json", "r"))
        test_ds = json.load(open("./dataset/combined_test.json", "r"))

    train_dataset = prepare_dataset(train_ds, char_encoder)
    val_dataset = prepare_dataset(val_ds, char_encoder)
    test_dataset = prepare_dataset(test_ds, char_encoder)

    return (
        data.DataLoader(train_dataset, batch_size=128),
        data.DataLoader(val_dataset, batch_size=128),
        data.DataLoader(test_dataset, batch_size=128),
    )


def run_experiment(dataset_name, config, num_epoch):
    train_dl, val_dl, test_dl = prepare_train_dataset(dataset_name)

    model = Seq2SeqTransformer(
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        n_head=config['num_head'],
        emb_dim=config['embedding_dim'],
        feedforward_dim=config['feedforward_dim'],
        input_vocab_size=char_encoder.get_total_char(),
        output_vocab_size=char_encoder.get_total_char(),
        dropout=config['dropout']
    )

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    scheduler = transformers.get_inverse_sqrt_schedule(
        optimizer,
        num_warmup_steps=4000
    )

    criterion = nn.CrossEntropyLoss(ignore_index=char_encoder.char2idx['<pad>'])
    model.to(device)

    patience = 0
    current_loss = 99

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": None
    }

    min_loss = 99
    best_epoch = 0

    for i in range(num_epoch):
        print(f"EPOCH : {i + 1}/{num_epoch}")
        train_loss = train_model(
            train_dl,
            model,
            criterion,
            optimizer,
            scheduler
        )

        val_loss = eval_model(
            val_dl,
            model,
            criterion
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch: {i+1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

        if val_loss > current_loss:
            patience += 1
        else:
            patience = 0

        if val_loss < min_loss:
            min_loss = val_loss
            best_epoch = i + 1
            print(f"Set best model on epoch {best_epoch}")
            best_model = model.state_dict().copy()
            
        
        current_loss = val_loss
        if patience >= 3:
            print(f"Early stop at epoch = {i + 1}")
            break
    
    test_loss = eval_model(
        test_dl,
        model,
        criterion
    )

    history['test_loss'] = test_loss

    return history, best_model, best_epoch


        


