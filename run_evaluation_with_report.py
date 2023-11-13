import os
import pandas as pd
import json
from utils import inference, metrics
import torch
import random
from tqdm import tqdm

random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model_dir = "./models/tuned/"

model_list = os.listdir(model_dir)
all_result = []
for i, item in enumerate(model_list):
    # sample name = tune-indocollex-layer-6-head-2-emb-64_epoch_60
    print(f"Evaluating {i+1} / {len(model_list)}")
    model_params = item.split("-")
    early_stop_epoch = int(item.split(".")[0].split("_")[-1])
    dataset_name = model_params[1]
    n_layer = int(model_params[3])
    n_head = int(model_params[5])
    emb_dim = int(model_params[7].split("_")[0])

    config = {
        "num_encoder_layers": n_layer,
        "num_decoder_layers": n_layer,
        "num_head": n_head,
        "embedding_dim": emb_dim,
        "feedforward_dim": 2048,
        "dropout": 0.1,
        "learning_rate": 0.005,
        "early_stop_epoch": early_stop_epoch
    }

    print(f"evaluating model {dataset_name} : ", config)

    model = inference.ModelInference(
        model_dir=f"{model_dir}{item}",
        device=device,
        config=config
    )

    if dataset_name == "indocollex":
        dataset_path = "./dataset/indo_collex_test.json"
    elif dataset_name == "col_id_norm":
        dataset_path = "./dataset/col_id_norm_test.json"

    datasets = json.load(open(dataset_path, "r"))

    true_list = []
    pred_list = []

    for item in tqdm(datasets):
        pred = model.normalize(item['slang'])
        true_list.append(item['formal'])
        pred_list.append(pred)

    result = metrics.evaluate(true_list, pred_list)

    print("result : ", result)
    config = config.update(result)
    all_result.append(config)

df_result = pd.DataFrame.from_records(all_result)
df_result.to_csv("./models/reports.csv", index=False)

