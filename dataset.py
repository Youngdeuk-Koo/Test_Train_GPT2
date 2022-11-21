import numpy as np
import re
from torch.utils.data import Dataset
from data_check import Chatbot_Data
from huggingface_load import load
from transformers import PreTrainedTokenizerFast
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logger = logging.getLogger('transformers.tokenization_utils_base')
logger.setLevel(logging.ERROR)

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'


koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
                                                                   bos_token=BOS, 
                                                                   eos_token=EOS, 
                                                                   unk_token="<unk>", 
                                                                   pad_token=PAD, 
                                                                   mask_token=MASK,
                                                                   )

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)
    
    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["Q"]
        q = re.sub(r"([?.!,])", r" ", q)
        
        a = turn["A"]
        a = re.sub(r"([?.!,])", r" ", a)
        
        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        
        #질문 길이가 최대 길이보다 클 경우
        if q_len > self.max_len:
            a_len = self.max_len - q_len                            # 답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:                                          # 질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]       # 질문길이를 최대길이의 으로
                q_len = len(q_toked)
                a_len = self.max_len - q_len                        # 답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            
        #질문 길이 + 답변 길이가 최대 길이보다 클 경우
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len                            # 답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:                                          # 질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]       # 질문길이를 최대길이의 반으로
                q_len = len(q_toked)
                a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
        
        # labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]    
        labels = [self.mask,] * q_len + a_toked[1:]
        
        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        
        # 답변 labels을 ids로 만든다
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        
        # 최대 길이만큼 패딩
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
            
        # 질문 + 답변을 ids로 만든다
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        
        # 최대 길이만큼 패딩
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
            
        # 질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)
    
# token_ids = + 질문문장 + + 감정 + + 답변 + + pad_token_id
# pad_token_id는 max_len에 일치
# mask는 질문q가 들어 가는 곳에는 0 답변 a가 위치한 곳에는 1 빈 공간에는 0으로
# labels는 질문 길이만큼 mask 그리도 답변 a의 id
            
        
