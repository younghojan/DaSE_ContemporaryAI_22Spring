import torch
import torch.nn as nn
from transformers import BertModel


class TxtEmbedding(nn.Module):
    def __init__(self, bert_model_path):
        super(TxtEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.bert(input_ids=x["txt_input_ids"],
                         attention_mask=x["txt_attention_mask"],
                         token_type_ids=x["txt_token_type_ids"])["pooler_output"]


class VisEmbedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(VisEmbedding, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class MultiModalClf(nn.Module):
    def __init__(self, bert_model_path, img_hidden_size, txt_hidden_size, num_classes):
        super(MultiModalClf, self).__init__()
        self.txt_transform = TxtEmbedding(bert_model_path)
        self.vis_transform = VisEmbedding(input_size=img_hidden_size, output_size=txt_hidden_size)
        self.fc = nn.Linear(txt_hidden_size * 2, num_classes)

    def forward(self, x):
        txt_embedding = self.txt_transform(x)
        vis_embedding = self.vis_transform(x["image_feature"])
        x = torch.cat([txt_embedding, vis_embedding], dim=1)
        out = self.fc(x)
        return out
