import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer


class BertPredict:
    # 初始化类
    def __init__(self, model_path, data_path):
        self.data_path = data_path
        self.model = BertForSequenceClassification.from_pretrained(model_path)  # 从 model_path 导入训练好的模型
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )

    # 读入测试数据
    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(line.replace('\n', '').split(', ', 1))
            dataset = pd.DataFrame(data[1:][:], columns=data[0])

        encodings = self.tokenizer(  # tokenize
            list(dataset['text']),
            truncation=True,
            padding='max_length',
            max_length=256
        )

        # 拼接成 Dataset 并且格式化为 torch 规范并能被 BERT 接受的数据集
        test_set = Dataset.from_dict(encodings)
        test_set.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask']
        )
        return test_set

    # 预测
    def predict(self):
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(self.load_data())
        preds = predictions.predictions.argmax(-1).tolist()
        map(str, preds)
        pred = [' ' + str(i) for i in preds]
        id = list(range(0, 2000))
        df = pd.DataFrame({'id': id, ' pred': pred})
        df.to_csv('submit_sample.txt', index=False, sep=',')  # 导出为 .txt 文件


if __name__ == '__main__':
    bert = BertPredict(model_path='best_model/', data_path='./dataset/test.txt')
    bert.predict()
