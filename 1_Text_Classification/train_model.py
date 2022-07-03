import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 加速训练


class BertTrain:
    # 初始化类
    def __init__(self, save_model_path, data_path):
        self.data_path = data_path  # 数据存放位置
        self.save_model_path = save_model_path  # 模型保存位置
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)  # 定义模型
        self.tokenizer = BertTokenizerFast.from_pretrained(  # 定义 tokenizer
            'bert-base-uncased',
            do_lower_case=True
        )

    # 读入训练数据
    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
            dataset = pd.DataFrame(data)

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=69)  # 使用分层抽样划分训练集和验证集
        texts = np.array(dataset.loc[:, 'raw'])
        labels = np.array(dataset.loc[:, 'label'])
        for train_index, val_index in split.split(texts, labels):
            train_texts, val_texts = texts[train_index], texts[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]

        # tokenize
        train_encodings = self.tokenizer(list(train_texts), truncation=True, padding='max_length', max_length=256)
        train_encodings['labels'] = train_labels

        val_encodings = self.tokenizer(list(val_texts), truncation=True, padding='max_length', max_length=256)
        val_encodings['labels'] = val_labels

        # 拼接成 Dataset 并且格式化为 torch 规范并能被 BERT 接受的数据集
        train_set = Dataset.from_dict(train_encodings)
        train_set.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        val_set = Dataset.from_dict(val_encodings)
        val_set.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        return train_set, val_set

    # 计算评价指标
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)  # 预测类别
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 训练模型
    def train(self):
        train_set, val_set = self.load_data()
        training_args = TrainingArguments(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            do_train=True,
            do_eval=False,
            no_cuda=False,
            load_best_model_at_end=True,
            learning_rate=3e-5,
            output_dir='./output',
            overwrite_output_dir=True,
            # weight_decay=1e-4,
            # warmup_ratio=0.1,
            # evaluation_strategy='steps',
            # eval_steps=400,
            # save_steps=400,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=self.compute_metrics()
        )
        trainer.train()
        trainer.save_model(self.save_model_path)


if __name__ == '__main__':
    bert = BertTrain(save_model_path='./model_train/', data_path='./dataset/train_data.txt')
    bert.train()
