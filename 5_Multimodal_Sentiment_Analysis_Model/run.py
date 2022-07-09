import argparse
import logging
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm

from config import Config
from models.int_fusion_model import MultiModalClf
from models.txt_model import TxtModelClf
from models.vis_model import VisualClf
from utils import set_seed, get_labels_and_frequencies, get_dataloaders, store_preds_to_disk

warnings.filterwarnings("ignore")

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def model_forward(batch, txt_model, multi_modal_model, vis_model):
    # text modality model
    txt_model.train()
    outputs_txt = txt_model(batch)

    # intermediate fusion model
    multi_modal_model.train()
    outputs_int = multi_modal_model(batch)

    # visual modality model
    vis_model.train()
    outputs_vis = vis_model(batch["image_feature"])

    return outputs_txt, outputs_int, outputs_vis


def eval(config, val_loader, txt_model, multi_modal_model, vis_model, criterion, store_preds=False):
    with torch.no_grad():
        losses, vote_preds, tgts = [[], [], []], [], []
        for batch in val_loader:
            batch = tuple(t.to(config.device) for t in batch)
            labels = batch[4]
            inputs = {
                "txt_input_ids": batch[0],
                "txt_attention_mask": batch[1],
                "txt_token_type_ids": batch[2],
                "image_feature": batch[3]
            }
            outputs_txt, outputs_int, outputs_vis = model_forward(inputs, txt_model, multi_modal_model, vis_model)

            loss = criterion(outputs_txt, labels)
            losses[0].append(loss.item())

            loss = criterion(outputs_int, labels)
            losses[1].append(loss.item())

            loss = criterion(outputs_vis, labels)
            losses[2].append(loss.item())

            outputs_txt = torch.nn.functional.softmax(outputs_txt, dim=1)
            outputs_int = torch.nn.functional.softmax(outputs_int, dim=1)
            outputs_vis = torch.nn.functional.softmax(outputs_vis, dim=1)

            vote_pred = torch.nn.functional.softmax(
                (outputs_txt * config.txt_vote_weight + outputs_int + outputs_vis * config.vis_vote_weight) / (
                        config.txt_vote_weight + 1 + config.vis_vote_weight), dim=1).argmax(
                dim=1).cpu().detach().numpy()
            vote_preds.append(vote_pred)

            tgt = labels.cpu().detach().numpy()
            tgts.append(tgt)

    tgts = [l for sl in tgts for l in sl]

    metrics = {"txt loss": np.mean(losses[0]), "int loss": np.mean(losses[1]), "vis loss": np.mean(losses[2])}

    vote_preds = [l for sl in vote_preds for l in sl]
    metrics["acc"] = accuracy_score(tgts, vote_preds)
    if store_preds:
        store_preds_to_disk(tgts, vote_preds, config)

    return metrics


def train(config, train_loader, val_loader, txt_model, multi_modal_model, vis_model):
    tr_total = max(config.txt_max_steps, config.int_max_steps, config.vis_max_steps)  # total(max) train steps
    config.num_train_epochs = tr_total // len(train_loader) + 1  # train epochs

    # Optimizer for text modality model
    txt_model_optimizer = torch.optim.Adam(
        [{"params": [p for n, p in txt_model.named_parameters() if "bert" in n],
          'lr': config.txt_bert_learning_rate},  # bert lr
         {"params": [p for n, p in txt_model.named_parameters() if "bert" not in n],
          'lr': config.txt_clf_learning_rate}]  # classifier lr
    )

    # Optimizer for intermediate fusion model
    multi_modal_model_optimizer = torch.optim.Adam(
        [{"params": [p for n, p in multi_modal_model.named_parameters() if "bert" in n],
          'lr': config.int_bert_learning_rate},  # bert lr
         {"params": [p for n, p in multi_modal_model.named_parameters() if "bert" not in n],
          'lr': config.int_clf_learning_rate}]  # classifier lr
    )

    # Optimizer for visual modality model
    vis_model_optimizer = torch.optim.Adam(
        [{"params": [p for n, p in vis_model.named_parameters()]}],
        lr=config.vis_learning_rate
    )

    loss_function = nn.CrossEntropyLoss()
    global_step = 0
    min_losses, best_acc, n_no_improve = [np.inf, np.inf, np.inf], 0, [0, 0, 0]

    train_iterator = trange(int(config.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")

        txt_model.train()
        multi_modal_model.train()
        vis_model.train()

        for step, batch in enumerate(epoch_iterator):
            losses = []
            batch = tuple(t.to(config.device) for t in batch)
            labels = batch[4]
            inputs = {
                "txt_input_ids": batch[0],
                "txt_attention_mask": batch[1],
                "txt_token_type_ids": batch[2],
                "image_feature": batch[3]
            }
            outputs_txt, outputs_int, outputs_vis = model_forward(inputs, txt_model, multi_modal_model, vis_model)

            loss = loss_function(outputs_txt, labels)
            losses.append(loss.item())
            txt_model_optimizer.zero_grad()
            loss.backward()
            txt_model_optimizer.step()

            loss = loss_function(outputs_int, labels)
            losses.append(loss.item())
            multi_modal_model_optimizer.zero_grad()
            loss.backward()
            multi_modal_model_optimizer.step()

            loss = loss_function(outputs_vis, labels)
            losses.append(loss.item())
            vis_model_optimizer.zero_grad()
            loss.backward()
            vis_model_optimizer.step()

            epoch_iterator.set_postfix({"txt loss": "{:.4f}".format(losses[0]),
                                        "int loss": "{:.4f}".format(losses[1]),
                                        "vis loss": "{:.4f}".format(losses[2])})

        txt_model.eval()
        multi_modal_model.eval()
        vis_model.eval()

        metrics = eval(config, val_loader, txt_model, multi_modal_model, vis_model, loss_function,
                       store_preds=config.store_preds)
        logger.info(
            "Eval: Text Model Loss: {:.5f} | Multimodal Model Loss: {:.5f} | Visual Model Loss: {:.5f} | Acc: {:.5f}".format(
                metrics["txt loss"], metrics["int loss"], metrics["vis loss"], metrics["acc"]))

        if best_acc < metrics["acc"]:
            torch.save(txt_model, 'best_text_model.pth')
            torch.save(multi_modal_model, 'best_multimodal_model.pth')
            torch.save(vis_model, 'best_visual_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory of the model predictions and checkpoints")
    parser.add_argument("--train_filepath", default=None, type=str, required=True,
                        help="The path to the train dataset which should be a .json file")
    parser.add_argument("--dev_filepath", default=None, type=str, required=True,
                        help="The path to the validation dataset which should be a .json file")
    parser.add_argument("--test_filepath", default=None, type=str, required=True,
                        help="The path to the test dataset which should be a .json file")

    # Other parameters
    parser.add_argument("--load_txt_model_path", default=None, type=str,
                        help="The path to previously trained text model")
    parser.add_argument("--load_int_model_path", default=None, type=str,
                        help="The path to previously trained multimodal model")
    parser.add_argument("--load_vis_model_path", default=None, type=str,
                        help="The path to previously trained visual model")
    parser.add_argument("--config_path", default="", type=str)

    parser.add_argument("--bert_model_path", default="bert-base-multilingual-cased", type=str)

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--store_preds", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--max_input_length", default=128, type=int)

    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)

    parser.add_argument("--img_hidden_size", default=2048, type=int)
    parser.add_argument("--txt_hidden_size", default=768, type=int)

    parser.add_argument("--int_bert_learning_rate", default=5e-5, type=float)
    parser.add_argument("--int_clf_learning_rate", default=1e-5, type=float)
    parser.add_argument("--int_max_steps", default=800, type=int)

    parser.add_argument("--vis_learning_rate", default=1e-5, type=float)
    parser.add_argument("--vis_max_steps", default=800, type=int)
    parser.add_argument("--vis_vote_weight", default=0.1, type=float)

    parser.add_argument("--txt_bert_learning_rate", default=5e-5, type=float)
    parser.add_argument("--txt_clf_learning_rate", default=1e-5, type=float)
    parser.add_argument("--txt_max_steps", default=800, type=int)
    parser.add_argument("--txt_vote_weight", default=0.1, type=float)

    parser.add_argument("--eval_steps", default=-1, type=int)
    parser.add_argument("--train_steps", default=-1, type=int)

    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument('--seed', type=int, default=129)

    # Print arguments
    args = parser.parse_args()
    logger.info(args)
    config = Config(vars(args))

    # Setup CUDA
    config.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")

    # Set seed
    set_seed(config)

    # Make directory if output_dir doesn't exist
    if os.path.exists(config.output_dir) is False:
        os.makedirs(config.output_dir)

    # Count num_labels and frequencies
    config.labels, config.label_freqs = get_labels_and_frequencies(config.train_filepath)
    config.num_labels = len(config.labels)
    logger.info(config.label_freqs)

    # Load datasets
    train_loader, eval_loader, test_loader = get_dataloaders(config)

    # Initialize models
    if config.load_vis_model_path is not None:
        vis_model = torch.load(config.load_vis_model_path).to(config.device)
    else:
        vis_model = VisualClf(
            config.img_hidden_size,
            config.num_labels
        ).to(config.device)

    if config.load_int_model_path is not None:
        multi_modal_model = torch.load(config.load_int_model_path).to(config.device)
    else:
        multi_modal_model = MultiModalClf(
            config.bert_model_path,
            config.img_hidden_size,
            config.txt_hidden_size,
            config.num_labels
        ).to(config.device)

    if config.load_txt_model_path is not None:
        txt_model = torch.load(config.load_txt_model_path).to(config.device)
    else:
        txt_model = TxtModelClf(
            config.bert_model_path,
            config.num_labels
        ).to(config.device)

    # Training!
    if config.do_train:
        train(config, train_loader, eval_loader, txt_model, multi_modal_model, vis_model)

    # Testing!
    if config.do_test:
        criterion = nn.CrossEntropyLoss()
        metrics = eval(config, test_loader, txt_model, multi_modal_model, vis_model, criterion,
                       store_preds=config.store_preds)
