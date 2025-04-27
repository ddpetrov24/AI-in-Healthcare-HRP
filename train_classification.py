import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .classification_dataset import load_data
from .metrics import AccuracyMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING GPU")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)
    

    model = load_model(model_name, **kwargs)
    print(model)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline="aug")
    val_data = load_data("classification_data/val", shuffle=False)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

    global_step = 0

    # training loop
    best_accuracy = 0
    train_acc = AccuracyMetric()
    val_acc = AccuracyMetric()
    num_epoch = 50
    for epoch in range(num_epoch):
        train_acc.reset()
        model.train()
        for img, label in train_data:
            #print(f"Image shape: {img.shape}, Label: {label}")
            img, label = img.to(device), label.to(device)

            pred = model(img)
            pred_classes = model.predict(img)
            loss_val = loss_func(pred,label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            logger.add_scalar('train/loss',loss_val.item(),global_step)
            train_acc.add(pred_classes,label)

            global_step += 1

        # torch.inference_mode calls model.eval() and disables gradient computation
        # total_accuracy = 0
        val_acc.reset()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred_classes = model.predict(img)
                val_acc.add(pred_classes,label)


        avg_validation_acc = val_acc.compute()['accuracy']
        if avg_validation_acc > best_accuracy:
            print("SAVING MODEL")
            best_accuracy = avg_validation_acc
            print(best_accuracy)
            print(epoch)
            save_model(model)

        # log average train and val accuracy to tensorboard
        #epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        #epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('train/acc',train_acc.compute()['accuracy'],global_step)
        logger.add_scalar('val/acc',val_acc.compute()['accuracy'],global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_acc.compute()['accuracy']:.4f} "
                f"val_acc={val_acc.compute()['accuracy']:.4f}"
            )
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
