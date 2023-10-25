import argparse
import torch
from utils import flow
import json


torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(
    prog="Indo Word Normalization",
    description="Training procedure implementation on Indonesian Word Normalization using Character level Seq2Seq transformers"
)

parser.add_argument("--dataset_name", required=True)
parser.add_argument("--num_epoch", required=True)
parser.add_argument("--config_name", required=True)
parser.add_argument("--model_name", required=True)

args = parser.parse_args()

config = json.load(open(f"./config/{args.config_name}.json", "r"))

history, best_model, best_epoch = flow.run_experiment(
    args.dataset_name,
    config,
    int(args.num_epoch)
)

json.dump(history, open(f"./logs/experiment_log_{args.model_name}.json", "w"))

torch.save(best_model, f"./models/{args.model_name}_epoch_{best_epoch}.pth")