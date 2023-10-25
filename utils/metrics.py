from torchmetrics.text import CharErrorRate, BLEUScore
import numpy as np


def evaluate(true_list, pred_list):
    assert len(true_list) == len(pred_list)
    acc = np.sum(np.array(true_list) == np.array(pred_list)).item() / len(pred_list)
    cer = CharErrorRate()
    bleu_1 = BLEUScore(n_gram=1)
    bleu_2 = BLEUScore(n_gram=2)
    cer_list = []
    bleu_1_list = []
    for true, pred in zip(true_list, pred_list):
        cer_list.append(cer(pred, true).item())
        bleu_1_list.append(bleu_1(" ".join(true), " ".join(pred)))

    return {
        "accuracy": acc,
        "char_error_rate": np.mean(cer_list).item(),
        "bleu_1": np.mean(bleu_1_list).item(),
    }
    