import argparse
import torch
from utils import flow
import json
import os

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(
    prog="Indo Word Normalization",
    description="Training procedure implementation on Indonesian Word Normalization using Character level Seq2Seq transformers"
)

parser.add_argument("--dataset_name", required=True)
parser.add_argument("--num_epoch", required=True)

args = parser.parse_args()

config = json.load(open(f"./config/tuning.json", "r"))

history_list = {}

model_list = os.listdir("./models/tuned")

for n_layer in config['num_encoder_decoder_layers']:
    for n_head in config['num_head']:
        for emb in config['embedding_dim']:
            model_name = f"tune-{args.dataset_name}-layer-{n_layer}-head-{n_head}-emb-{emb}"

            for item in model_list:
                if model_name in item:
                    is_found = True
                    print()
                else:
                    is_found = False

            config_item = {
                "num_encoder_layers": n_layer,
                "num_decoder_layers": n_layer,
                "num_head": n_head,
                "embedding_dim": emb,
                "feedforward_dim": config['feedforward_dim'],
                "dropout": config['dropout'],
                "learning_rate": config['learning_rate']
            }
            for item in model_list:
                if model_name in item:
                    is_found = True
                    print("Config ", config_item, "found")
                    break
                else:
                    is_found = False
            
            if not is_found:
                print("Tuning : ", config_item)
                history, best_model, best_epoch = flow.run_experiment(
                    args.dataset_name,
                    config_item,
                    int(args.num_epoch)
                )

                json.dump(history, open(f"./logs/experiment_log_tuning_{model_name}.json", "w"))

                torch.save(best_model, f"./models/tuned/{model_name}_epoch_{best_epoch}.pth")
            else:
                continue