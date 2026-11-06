import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import precision_recall_fscore_support

from sst2 import CLASSES
from DiegoShan import smlp2


def fold_metrics(run, folds):
    """Accuracy and macro precision, recall and F1 per fold, from a run predictions."""
    metrics = {"accuracy": {}, "precision": {}, "recall": {}, "f1": {}}

    for fold, split in folds.items():
        gold = [label for _, label in split["validation"]]
        pred = run["predictions"][fold]
        p, r, f1, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)

        metrics["accuracy"][fold] = float(run["accuracies"][fold])
        metrics["precision"][fold] = float(p)
        metrics["recall"][fold] = float(r)
        metrics["f1"][fold] = float(f1)

    return metrics


def mlp2_cross_validation(folds, encoder_choice, tokenizer, save_to, hidden_dim2=64, dropout=0.3, num_epochs=5, **mlp2_args):
    result = {"models": {}, "predictions": {}, "accuracies": {}}

    for fold, split in folds.items():
        encoder = encoder_choice(tokenizer)
        if hasattr(encoder, "fit"):
            encoder.fit([text for text, _ in split["training"]])

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True)

        metrics, best_epoch, predictions = smlp2.train_neural_network_model(
            split["training"], split["validation"], len(CLASSES), encoder,
            save_to=f"{fold_folder}/model", hidden_dim2=hidden_dim2,
            dropout=dropout, num_epochs=num_epochs, **mlp2_args,
        )

        result["models"][fold] = f"{fold_folder}/model"
        result["predictions"][fold] = predictions
        result["accuracies"][fold] = metrics[best_epoch]["val_metrics"]["accuracy"]

    return result
