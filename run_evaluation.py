import argparse
import torch
from utils import metrics, inference
import json
from tqdm import tqdm
import random

random.seed(42)


torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(
    prog="Indo Word Normalization",
    description="Evaluation procedure implementation on Indonesian Word Normalization using Character level Seq2Seq transformers"
)

parser.add_argument("--dataset_path", required=True)
parser.add_argument("--config_name", required=True)
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

true_list = []
pred_list = []

config = json.load(open(f"./config/{args.config_name}.json", "r"))

model = inference.ModelInference(
    model_dir=args.model_path,
    device=device,
    config=config
)

datasets = json.load(open(args.dataset_path, "r"))

for item in tqdm(datasets):
    pred = model.normalize(item['slang'])
    true_list.append(item['formal'])
    pred_list.append(pred)

print(metrics.evaluate(true_list, pred_list))

print("Prediction sample")
selected_dataset = random.choices(datasets, k=10)

for item in selected_dataset:
    pred = model.normalize(item['slang'])
    print(f"True sample {item['slang']} >> {item['formal']}")
    print(f"Pred sample {item['slang']} >> {pred}")
