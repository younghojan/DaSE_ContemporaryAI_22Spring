import torch
import torch.nn as nn
from transformers import BertModel


class TxtModelClf(nn.Module):
    def __init__(self, model_name_or_path, num_classes):
        super(TxtModelClf, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = torch.nn.Linear(768, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        output = self.bert(input_ids=x["txt_input_ids"], attention_mask=x["txt_attention_mask"],
                           token_type_ids=x["txt_token_type_ids"])
        output = output['pooler_output']
        output = self.linear(output)
        output = self.dropout(output)

        return output
