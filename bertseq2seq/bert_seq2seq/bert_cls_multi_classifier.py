## bert encoder模型
from multiprocessing import pool
import torch
import torch.nn as nn
from trainsum.bertseq2seq.bert_seq2seq.tokenizer import Tokenizer
from trainsum.bertseq2seq.bert_seq2seq.basic_bert import BasicBert

class BertClsMultiClassifier(BasicBert):
    """
    """
    def __init__(self, word2ix, target_size, model_name="roberta"):
        super(BertClsMultiClassifier, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)
    
    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        # predictions = torch.sigmoid(predictions)
        batch_size = predictions.shape[0]
        # predictions = predictions.view(-1)
        # labels = labels.view(-1)
        loss = nn.BCEWithLogitsLoss(reduction="none")
        return loss(predictions, labels).sum() / batch_size
    
    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        
        text = text.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        all_layers, pooled_out = self.bert(text, 
                                    output_all_encoded_layers=True)

        if use_layer_num != -1:
            pooled_out = all_layers[use_layer_num][:, 0]

        predictions = self.final_dense(pooled_out)

        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss 
        else :
            return predictions
