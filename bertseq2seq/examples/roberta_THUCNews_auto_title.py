# -*- coding:utf-8 -*-
## THUCNews 原始数据
import torch 
from tqdm import tqdm
import time
import glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

vocab_path = r"E:\Pycharm_projects_zjy\2-testsum\12-12-768-16\vocab.txt"  # roberta模型字典的位置
word2idx = load_chinese_base_vocab(vocab_path)
model_name = "bert"  # 选择模型名字
model_path = r"E:\Pycharm_projects_zjy\2-testsum\12-12-768-16\pytorch_model.bin"  # 模型位置
#recent_model_path = r"E:\Pycharm_projects_zjy\2-testsum\bert_seq2seq-master\state_dict\bert_auto_title_model.bin"   # 用于把已经训练好的模型继续训练
model_save_path = r"E:\Pycharm_projects_zjy\2-testsum\bert_seq2seq-master\state_dict\bert_ti_title_model.bin"
batch_size = 4
lr = 1e-5
maxlen = 256

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self) :
        ## 一般init函数是加载所有数据(⊙o⊙)…
        super(BertDataset, self).__init__()
        ## 拿到所有文件名字
        self.txts = glob.glob(r'E:\Pycharm_projects_zjy\2-testsum\Ti_summ_data\测试\*.txt')
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        #print(i)

        text_name = self.txts[i]
        #print(self.txts)
        with open(text_name, "r", encoding="utf-8") as f:
            text = f.read()
            #print("text_0:", text)
        text = text.split('\n')
        #print("text:", text)
        if len(text) > 1:
            title = text[0]
            content = '\n'.join(text[1:])
            #print("content: ", content)
            token_ids, token_type_ids = self.tokenizer.encode(
                content, title, max_length=maxlen
            )
            output = {
                "token_ids": token_ids,
                "token_type_ids": token_type_ids,
            }
            #print("output: ", output)
            return output

        return self.__getitem__(i + 1)

    def __len__(self):

        return len(self.txts)
        
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        #print("pad_indice: ", pad_indice)
        return torch.tensor(pad_indice)
    # for data in batch:
    #     print("data: ", data)
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    #print("max_length: ", max_length)
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    #print("token_ids_padded: ", token_ids_padded)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()
    #print("target_ids_padded: ", target_ids_padded)

    return token_ids_padded, token_type_ids_padded, target_ids_padded
import json
from rouge import Rouge

class Trainer:
    def __init__(self):
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        ## 加载预训练的模型参数～
        
        self.bert_model.load_pretrain_params(model_path)
        # 加载已经训练好的模型，继续训练

        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset()
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        print("start train")
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True):
            step += 1
            if step % 5 == 0:  #1000
                self.bert_model.eval()
                # test_data = [" ཟླ་3ཚེས་15ཉིན། ལྗོངས་ཏང་ཨུད་རྩ་འཛུགས་པུས་པུའུ་དོན་ཚོགས་འདུ་འཚོགས་ནས་སྤྱི་ཁྱབ་ཧྲུའུ་ཅི་ཞི་ཅིན་ཕིང་གིས་ཀྲུང་དབྱང་ཏང་གི་སློབ་གྲྭའི(རྒྱལ་ཁབ་སྲིད་འཛིན་སློབ་གླིང)གཞོན་དར་ལས་བྱེད་པའི་གསོ་སྦྱོང་འཛིན་གྲྭའི་འགོ་ཚུགས་མཛད་སྒོར་གནང་བའི་གསུང་བཤད་གལ་ཆེན་གྱི་དགོངས་དོན་བརྒྱུད་བསྒྲགས་དང་སློབ་སྦྱོང་དང་ལག་བསྟར་དོན་འཁྱོལ་བྱེད་ཐབས་ལ་ཞིབ་འཇུག་བྱས་པ་རེད། ལྗོངས་ཏང་ཨུད་ཀྱི་རྒྱུན་ཨུ་རྩ་འཛུགས་པུའི་པུའུ་ཀྲང་ལའེ་ཅའོ་ཡིས་ཚོགས་འདུ་གཙོ་སྐྱོང་གནང་བ་དང་འབྲེལ་གསུང་བཤད་ཀྱང་གནང་བ་རེད།ལའེ་ཅའོ་ཡིས་བསྟན་དོན། 2019ལོ་ཚུན་དུ་སྤྱི་ཁྱབ་ཧྲུའུ་ཅི་ཞི་ཅིན་ཕིང་གིས་ཀྲུང་དབྱང་ཏང་གི་སློབ་གྲྭའི(རྒྱལ་ཁབ་སྲིད་འཛིན་སློབ་གླིང)གཞོན་དར་འཛིན་གྲྭའི་སློབ་འགོ་ཚུགས་པའི་སློབ་ཚན་དང་པོ་སྐུ་ངོ་མས་ཐེངས་6འཚོགས་གནང་ཞིང་། དེ་ལས་ཁོང་གིས་ན་གཞོན་ལས་བྱེད་པར་ཐུགས་ཁུར་དང་གཅེས་སྐྱོང་དང་། རྒྱབ་སྐྱོར་ཆེན་པོ། མངགས་བཅོལ་ཟབ་མོ་བཅས་གནང་ལུགས་མཚོན་ཡོད། རྩ་འཛུགས་སྡེ་ཚན་གྱིས་སྤྱི་ཁྱབ་ཧྲུའུ་ཅི་ཞི་ཅིན་ཕིང་གི་གསུང་བཤད་གལ་ཆེན་གྱི་དགོངས་དོན་སློབ་སྦྱོང་དང་གོ་བ་གཏིང་ཟབ་བླངས་ཏེ། ན་གཞོན་ལས་བྱེད་པས་ལས་དོན་ཡག་པོ་སྒྲུབ་པའི་ཆབ་སྲིད་ཀྱི་འགན་འཁྲི་འཁུར་སེམས་དང་ལོ་རྒྱུས་ཀྱི་ལས་འགན་སྒྲུབ་སེམས་ཆེ་རུ་ཏན་ཏིག་གཏོང་དགོས་པ་དང་། སྤྱི་ཁྱབ་ཧྲུའུ་ཅི་ཞི་ཅིན་ཕིང་གི་གསུང་བཤད་གལ་ཆེན་ཐེངས་6གི་དགོངས་དོན་དང་ཟུང་འབྲེལ་བྱས་ནས་གཅིག་གྱུར་གྱིས་སློབ་སྦྱོང་དང་དོན་འཁྱོལ་བྱས་ཏེ་ཕུལ་བྱུང་ན་གཞོན་ལས་བྱེད་པ་གསོ་སྐྱོང་དང་། འདེམས་སྒྲུག དོ་དམ། བེད་སྤྱོད་བཅས་ཀྱི་ལས་དོན་ཏན་ཏིག་དང་ཡག་པོ་སྒྲུབ་དགོས། སྤྱོད་ཚུལ་ལེགས་གཏོང་དང་དོན་འཁྱོལ་ལ་དམ་འཛིན་ནན་པོ་བྱེད་པའི་ལས་དོན་དང་ཟུང་འབྲེལ་བྱས་ནས་ན་གཞོན་ལས་བྱེད་པས“དོན་ཆེན་བཞི”དང“འགན་ལེན་བཞི”ལ་དམིགས་རྒྱུར་སྐུལ་འདེད་གཏོང་བ་དང“འཛུགས་གཏོད་བཞི”ལ་སྐུལ་འདེད་གཏོང་བར་དམིགས་ནས“མདུན་གྲལ་དུ་སླེབས་པ་བཞི”ཐུབ་པ་བྱས་ཏེ། དཀའ་སྤྱད་སྙིང་རུས་ཀྱིས་ལས་དོན་སྒྲུབ་པ་དང་བློ་སྟོབས་ཆེར་སྐྱེད་དང་བྱ་སྤྱོད་དངོས་ལ་བརྟེན་ནས་ཏང་གི་ཚོགས་ཆེན་ཉི་ཤུ་པ་རྒྱལ་ཁའི་ངང་འཚོག་རྒྱུར་བསུ་བ་ཞུ་དགོས།",
                #  "ཉེ་ཆར། ལྗོངས་ཏང་ཨུད་ཀྱི་རྒྱུན་ཨུ་དང་། རང་སྐྱོང་ལྗོངས་ཀྱི་རྒྱུན་ལས་ཀྲུའུ་ཞི་གཞོན་པ་ཤའོ་ཡིའུ་ཚའེ་ཡིས་རང་སྐྱོང་ལྗོངས་ཀྱི་ས་མཐོའི་ཁྱད་ལྡན་ཞིང་འབྲོག་ཐོན་ལས་ཆེད་དོན་ཚོགས་ཆུང་གི་ཆེད་དོན་ཚོགས་འདུ་གཙོ་སྐྱོང་གནང་བ་རེད། ཤའོ་ཡིའུ་ཚའེ་ཡིས། གཅིག་ནས་ཚུགས་གནས་མཐོ་རུ་བཏང་སྟེ། ས་མཐོའི་ཁྱད་ལྡན་ཞིང་འབྲོག་ཐོན་ལས་སྤུས་ཚད་མཐོ་པོས་འཕེལ་རྒྱས་འགྲོ་རྒྱུར་སྐུལ་འདེད་མགྱོགས་མྱུར་གཏོང་བ་དེ་ཞིང་འབྲོག་མང་ཚོགས་ཀྱི་ཡོང་འབབ་མང་དུ་གཏོང་བ་དང་ཞིང་འབྲོག་པའི་འཚོ་བའི་ཆུ་ཚད་མཐོ་རུ་གཏོང་རྒྱུར་དོན་སྙིང་གལ་ཆེན་ལྡན་པར་གོ་བ་གཏིང་ཟབ་ལེན་དགོས། ཐོན་ལས་དང་པོ་ལེགས་སུ་གཏོང་བ་དང་། ཐོན་ལས་གཉིས་པ་རྒྱ་སྟོབས་ཆེ་རུ་གཏོང་བ། ཐོན་ལས་གསུམ་པའི་ཆུ་ཚད་མཐོ་རུ་གཏོང་བ་བཅས་ཀྱི་དམིགས་ཚད་བླང་བྱ་གཞིར་བཟུང་། ཐོན་ཁུངས་ཀྱི་རྒྱུ་རྐྱེན་གཙོ་བོ་གང་ལེགས་སྤྱོད་པ་དང་། ཡུལ་བབ་དང་བསྟུན་ནས་ཞིང་འབྲོག་ཐོན་ལས་རྒྱ་སྟོབས་ཆེ་རུ་བཏང་སྟེ་གཙོ་གནད་དང་འགག་རྩའི་ཁྱབ་ཁོངས་ལ་ཚད་བརྒལ་ཡོང་བ་བྱེད་དགོས། གཉིས་ནས་ལས་དོན་ཏན་ཏིག་སྤེལ་ཏེ། ནས་དང་། འབྲི་གཡག ལུག་སོགས་ཁྱད་ལྡན་ཞིང་འབྲོག་ཐོན་ལས་གཙོ་གནད་དུ་བཟུང་ནས་ཁྱད་ལྡན་ཞིང་འབྲོག་ལས་དང་ནགས་ཁྲོད་ཐོན་ཁུངས་གསར་སྤེལ་དང་། སྐྱེ་ཁམས་དམངས་ཕྱུག་ཐོན་ལས་ཅན་འཛུགས་སྐྲུན་བྱེད་མགྱོགས་སུ་འགྲོ་རྒྱུར་སྐུལ་འདེད་གཏོང་དགོས། གསུམ་ནས་སྐུལ་འདེད་དེ་བས་མགྱོགས་པོ་བཏང་སྟེ། འཕེལ་རྒྱས་ཀྱི་བླང་བྱར་དམ་པོར་དམིགས་པ་དང་། ལས་འགན་རེ་རེ་བཞིན་ཞིབ་ཚགས་སུ་གཏོང་བ། གཙོ་གནད་རྣམ་གྲངས་དང་གཙོ་གནད་ལས་དོན་ལ་འགན་འཁྲི་ཁ་གསལ་གྱི་ལས་ཀའི་འཁོར་སྦྲེལ་ཆགས་པ་བྱེད་པ། ལས་དོན་ཁག་དུས་ཚོད་དང་འཆར་གཞི་གཞིར་བཟུང་གོ་རིམ་ལྡན་པའི་ངང་སྐུལ་འདེད་གཏོང་བ་བཅས་བྱས་ཏེ། དམིགས་ཚད་ལས་འགན་ཁག་གང་མགྱོགས་དོན་འཁྱོལ་ཐུབ་པའི་འགན་ལེན་བྱེད་དགོས། བཞི་ནས་སྤུས་ཚད་མཐོ་རུ་བཏང་སྟེ། རྣམ་གྲངས་ཕྱོགས་སྒྱུར་རིམ་སྤོར་བྱེད་ཤུགས་ཆེ་རུ་བཏང་ནས་ལས་ཕྱོད་ཆེ་རུ་གཏོང་བ་དང་། མ་དངུལ་ཁག་སྤྱོད་གོ་ཆོད་སར་སྤྱོད་པ། བསྐྲུན་ཟིན་པའི་རྣམ་གྲངས་དག་ལ་དུས་ཐོག་ཏུ་རྩིས་ལེན་ལས་དོན་སྒྲིག་འཛུགས་བྱེད་པ། རྣམ་གྲངས་གཅིག་བསྐྲུན་མ་ཐག་དེ་རྩིས་ལེན་ཐུབ་པའི་བརྩོན་ལེན་བྱེད་པ་བཅས་ཀྱི་ཐོག་ནས་འཛུགས་སྐྲུན་རྣམ་གྲངས་ལས་ཕྱོད་ཆེན་པོ་དང་སྤུས་ཚད་མཐོ་པོ་ཡོང་བའི་འགན་ལེན་བྱེད་དགོས་པའི་ནན་བཤད་གནང་བ་རེད།",
                #  "དབུལ་སྒྲོལ་འགག་སྒྲོལ་མཁྲེགས་སར་གཏུགས་ཤིང་འགག་སྒྲོལ་གྱི་རྒྱལ་ཁ་ལེན་པའི་མཐའ་མའི་གདོང་མཆོང་གི་དུས་སུ་སླེབས་པས། རང་སྐྱོང་ལྗོངས་ཏང་ཨུད་ཀྱིས་དེར་ཚད་མཐོའི་མཐོང་ཆེན་གནང་གི་ཡོད། རང་སྐྱོང་ལྗོངས་ཏང་ཨུད་ཀྱི་ལས་དོན་གྱི་བཀོད་སྒྲིག་གཞིར་བཟུང་། ཟླ་2ཚེས་19ཉིན་ཐོག་མཐའ་བར་གསུམ་དུ་དབུལ་སྒྲོལ་གྱི་ཇུས་ཆེན་ཐུགས་སུ་འཆང་བའི་རང་སྐྱོང་ལྗོངས་ཏང་ཨུད་ཀྱི་ཧྲུའུ་ཅི་ཝུའུ་དབྱིང་ཅེས་ལྗོངས་ཡོངས་ཀྱི་ཐག་རིང་ཤོས་མངའ་རིས་ས་ཁུལ་གྱི་མཚོ་ཆེན་དང་སྒེར་རྩེ་སོགས་རྫོང་དུ་ཕེབས་ཏེ་རྒྱ་ཆེའི་གཞི་རིམ་ལས་བྱེད་པ་དང་མང་ཚོགས་ལ་གཟིགས་ནས་འཚམས་འདྲི་གནང་བ་དང་། རྫོང་ཁག་གིས་དབུལ་སྐྱོར་གསར་སྤེལ་སྐོར་གྱི་སྤྱི་ཁྱབ་ཧྲུའུ་ཅི་ཞི་ཅིན་ཕིང་གི་རྣམ་བཤད་གལ་ཆེན་ལག་བསྟར་དོན་འཁྱོལ་གཏིང་ཟབ་ལ་ལྟ་སྐུལ་མཛུབ་ཁྲིད་དང་། གཙོ་གནད་ཐོག་གནད་སྨིན་དབུལ་སྐྱོར་དང་གནད་སྨིན་དབུལ་སྒྲོལ་གྱི་ལས་དོན་ལ་གནས་པའི་གནད་དོན་བཙལ་འཚོལ་དང་ཐག་གཅོད། ས་གནས་ས་ཐོག་ཏུ་ཀྲུང་དབྱང་གི་མཛད་ཇུས་གལ་ཆེན་དང་ལྗོངས་ཏང་ཨུད་ཀྱི་ཐབས་ཇུས་བཀོད་སྒྲིག་དོན་འཁྱོལ་གནད་སྨིན་ལ་སྐུལ་འདེད་བཏང་སྟེ། ཤིན་ཏུ་དབུལ་བའི་རྫོང་གིས་དུས་བཀག་ལྟར་དབུལ་ཞྭ་ཕུད་ཆེད་སྒུལ་ཤུགས་ཆེན་པོ་བསྣན་པ་རེད། ཝུའུ་དབྱིང་ཅེས་དབུལ་སྒྲོལ་འགག་སྒྲོལ་གྱི་དམག་འཐབ་ལ་རྒྱལ་ཁ་ལེན་པ་ནི། ཀྲུང་དབྱང་གིས་སྤྱི་ཡོངས་ལ་དམིགས་ནས་བཏོན་གནང་བའི་ཐབས་ཇུས་བཀོད་སྒྲིག་གལ་ཆེན་ཞིག་ཡིན་པ་དང་། ཕྱོགས་ཡོངས་ནས་འབྱོར་འབྲིང་སྤྱི་ཚོགས་སྐྲུན་པའི་དཀའ་ཚེགས་ཆེ་ཤོས་ཀྱི་ལས་འགན་ཞིག་རེད། ད་ལོ་ལྗོངས་ཡོངས་སུ་རྫོང་19དབུལ་སྒྲོལ་ཞྭ་འཕུད་བྱེད་དགོས་པས། ལྗོངས་ཡོངས་ཀྱི་མི་རིགས་ཁག་གི་ལས་བྱེད་པ་དང་མང་ཚོགས་ཚོས་འགན་འཁྲི་ཕྲག་ཏུ་འཁུར་ནས་དཀའ་སྤྱད་དངོས་སྒྲུབ་དང་། བརྩེ་བ་བཅངས་ནས་འབད་འཐབ་དང་ལེགས་སྐྱེས་འབུལ་བ། གནད་སྨིན་ཤུགས་འདོན་ལ་འབད་བརྩོན་ལྷོད་མེད། དབུལ་པོའི་རྩ་བ་འབྱིན་པ་བཅས་བྱས་ཏེ་ནུས་ཤུགས་ཡོད་རྒུས་དབུལ་སྒྲོལ་འགག་སྒྲོལ་གྱི་དམག་འཐབ་ལ་རྒྱལ་ཁ་ལེན་དགོས། རིམ་ཁག་འགོ་ཁྲིད་ལས་བྱེད་པས་དབུལ་སྒྲོལ་འགག་སྒྲོལ་གྱི་ལྟ་སྐུལ་མཛུབ་ཁྲིད་ཀྱི་ལས་དོན་ལ་ཚད་མཐོའི་མཐོང་ཆེན་བྱས་ཏེ། གོང་དཔེ་འོག་སྟོན་དང་སྣེ་ཁྲིད་ནས་ཐག་རིང་ལུང་ཁུག་དང་དཀའ་ཚེགས་ཆེ་བའི་ས་ཁུལ་དུ་ལས་དོན་ལ་ཞིབ་བཤེར་མཛུབ་ཁྲིད་དང་སྐུལ་འདེད་གཏོང་བ་མཐའ་འཁྱོངས་བྱས་ཏེ་དུས་བཀག་ལྟར་ཚང་མ་དབུལ་ཞྭ་ཕུད་ཐུབ་པ་བྱེད་དགོས་པའི་ནན་བཤད་གནང་བ་རེད། མཚོ་ཆེན་རྫོང་དང་སྒེར་རྩེ་རྫོང་གི་ས་བབ་མཐོ་བ་དང་། ཆ་རྐྱེན་ཞན་པ། མི་འབོར་དབུལ་པོ་མང་བས་ཧ་ཅང་དབུལ་པོ་ཞིག་རེད། ད་ལོ་དབུལ་ཞྭ་ཕུད་རྩིས་ཡོད་སྟབས། མིག་སྔར་རྫོང་དེ་གཉིས་ཀྱིས་སྟོབས་ཤུགས་གཅིག་སྡུད་ཀྱིས་གནད་སྨིན་དབུལ་སྐྱོར་དང་གནད་སྨིན་དབུལ་སྒྲོལ་གྱི་ལས་དོན་ཁག་ལ་དམ་འཛིན་ཡག་པོ་བྱས་ནས་དབུལ་སྒྲོལ་ཞྭ་འཕུད་ཀྱི་ལས་འགན་ལེགས་གྲུབ་ཡོང་བའི་འགན་ལེན་བྱེད་བཞིན་ཡོད་པ་རེད། མཚོ་ཆེན་རྫོང་དྲ་ཐོག་ཚོང་དོན་ཕྱོགས་བསྡུས་ཐོག་ཁང་འབབ་འཕར་གྱི་བཀྲག་མདངས་འཚེར་ས་གསར་པ་ཞིག་ཏུ་གྱུར་པ་དང་། སྒེར་རྩེ་རྫོང་དུང་མཚོ་ཤང་གི་སློབ་ཆེན་སློབ་མའི་ལས་གཏོད་ཇ་ཁང་གི་ཚོང་ཡག་པ། འོ་མ་ཤང་འཆམ་གུག་གྲོང་ཚོའི་ཆེད་ལས་མཉམ་ལས་ཁང་གིས་མང་ཚོགས་དབུལ་པོ་སྔར་ལས་མང་བའི་འབབ་འཕར་ལ་སྐུལ་ཁྲིད་བྱས་པ་སོགས་ཕྱོགས་མང་པོའི་རོགས་རམ་དང་རྒྱབ་སྐྱོར་འོག་དེ་སྔ་དབུལ་ཕོངས་རྗེས་ལུས་ཀྱི་ཐག་རིང་རི་གྲོང་དག་ལ་དགའ་འོས་པའི་འགྱུར་ལྡོག་འགྲོ་བཞིན་ཡོད།"]
                # for text in test_data:
                #
                #     print(self.bert_model.generate(text, beam_size=3))
                rouge = Rouge()
                test_file = glob.glob(r"E:\Pycharm_projects_zjy\2-testsum\Ti_summ_data\测试\test\*.json")
                num_file = len(test_file)
                rouge_1_item = [0.0, 0.0, 0.0]
                #with open("./auto_title_res.txt", "a+", encoding="UTF-8") as fw:
                for s_file in test_file:
                    with open(s_file, "r", encoding="UTF-8") as f:
                        c = f.read()
                        # print("c: ", c)
                        j = json.loads(c)
                        # print("j: ", len(j))
                        title = j["abstract"]
                        text = j["Title"]
                        # print("text: ", text)
                        out = self.bert_model.generate(text, beam_size=3)
                        # print("out: ", out)
                        #fw.write(title + "\t" + out + "\t" + str(text) + "\n")

                        rouge_score = rouge.get_scores(out, title, avg=True) #(预测，参考)
                        print("rouge_score: ",rouge_score)
                        rouge_1 = rouge_score[0]["rouge-1"]
                        rouge_1_item[0] += rouge_1["f"]
                        rouge_1_item[1] += rouge_1["p"]
                        rouge_1_item[2] += rouge_1["r"]
                        # print(rouge_score[0]["rouge-2"])
                        # print(rouge_score[0]["rouge-l"])
                for i in range(len(rouge_1_item)):
                    rouge_1_item[i] = rouge_1_item[i] / num_file

                print("rouge_1_item: ",rouge_1_item)

                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 8000== 0: #8000
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                               
                                                )
            report_loss += loss.item()
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(model_save_path)

if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 1

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)