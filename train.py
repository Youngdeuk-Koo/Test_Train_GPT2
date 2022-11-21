import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm

from data_check import Chatbot_Data
from huggingface_load import load
from dataset import ChatbotDataset
from collate_batch import collate_batch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'

Chatbot_Data = Chatbot_Data()
model = load.model_load()
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# device = torch.device("cuda")
print(device)
train_set = ChatbotDataset(Chatbot_Data, max_len=40,)
#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_set, 
                              batch_size=32, 
                              num_workers=0, 
                              shuffle=True, 
                              collate_fn=collate_batch,)

model.to(device)
model.train()

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
Sneg = -1e18

print('Start')
for epoch in range(epochs):
    for batch_idx, samples in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad(0)
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits                        # 입력 요소의 로짓히 포함된 새 tensor를 반환
        
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        
        # 학습 끝
        optimizer.step()
print('end')

PATH = './SAVE_MODEL_DIR/'
torch.save(model, PATH + "model")
print('model save success')