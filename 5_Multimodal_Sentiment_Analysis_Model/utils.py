import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import transforms
from transformers import AutoTokenizer


# Read data from .json file
class JsonDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length,
                 transforms,
                 labels):
        self.data = json.load(open(data_path))
        self.img_feature_extractor = ImageFeatureExtractor()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms
        self.labels = labels
        self.n_classes = len(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index]["text"]
        txt_encoding = self.tokenizer(sentence, max_length=self.max_seq_length, truncation=True, padding='max_length',
                                      add_special_tokens=True)  # tokenize once, used separately in text model and multimodal model

        label = torch.tensor([0])
        label[0] = self.labels.index(self.data[index]["label"])

        image = Image.open(self.data[index]["img"]).convert("RGB")  # convert the image to three channels(RGB)
        image = Variable(torch.unsqueeze(self.transforms(image), dim=0).float(), requires_grad=False)
        img_feature = torch.flatten(self.img_feature_extractor(image))
        return {
            "txt_encoding": txt_encoding,  # (input_ids, attention_mask, token_type_ids)
            "image_feature": img_feature,  # 2048-D
            "label": label
        }


# Extract image feature using ResNet152(discarding its final FC layer)
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-1])
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.extractor(x)


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_dataloaders(config):
    transforms = get_image_transforms()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path, do_lower_case=False)

    train_dataset = JsonDataset(
        data_path=config.train_filepath,
        tokenizer=tokenizer,
        max_seq_length=config.max_input_length,
        transforms=transforms,
        labels=config.labels
    )

    eval_dataset = JsonDataset(
        data_path=config.dev_filepath,
        tokenizer=tokenizer,
        max_seq_length=config.max_input_length,
        transforms=transforms,
        labels=config.labels
    )

    test_dataset = JsonDataset(
        data_path=config.test_filepath,
        tokenizer=tokenizer,
        max_seq_length=config.max_input_length,
        transforms=transforms,
        labels=config.labels
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.train_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        sampler=RandomSampler(eval_dataset),
        batch_size=config.eval_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=config.eval_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    return train_dataloader, eval_dataloader, test_dataloader


def collate_fn(batch):
    lens = [len(row["txt_encoding"].input_ids) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    tokentype_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = torch.LongTensor(input_row["txt_encoding"].input_ids)
        mask_tensor[i_batch, :length] = torch.LongTensor(input_row["txt_encoding"].attention_mask)
        tokentype_tensor[i_batch, :length] = torch.LongTensor(input_row["txt_encoding"].token_type_ids)

    img_tensor = torch.stack([row["image_feature"] for row in batch])
    tgt_tensor = torch.cat([row["label"] for row in batch])

    return text_tensor, mask_tensor, tokentype_tensor, img_tensor, tgt_tensor


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [line["label"] for line in json.load(open(path))]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


def store_preds_to_disk(tgts, preds, config):
    with open(os.path.join(config.output_dir, "test_labels_pred.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in preds]))
    with open(os.path.join(config.output_dir, "test_labels_gold.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in tgts]))
    with open(os.path.join(config.output_dir, "test_labels.txt"), "w") as fw:
        fw.write(" ".join([str(l) for l in config.labels]))


def save_checkpoint(model_name, state, path):
    filename = os.path.join(path, model_name + ".pt")
    torch.save(state, filename)
