import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

def model_load():

    tokenizer = PreTrainedTokenizerFast.from_pretrained("SAVE_TOKENIZER_DIR", 
                                                                   bos_token=BOS, 
                                                                   eos_token=EOS, 
                                                                   unk_token="<unk>", 
                                                                   pad_token=PAD, 
                                                                   mask_token=MASK,
                                                                   )

    # model = GPT2LMHeadModel.from_pretrained('SAVE_MODEL_DIR/model')
    model = torch.load("SAVE_MODEL_DIR/model", map_location=device)

    return model, tokenizer

def gpt_chatbot(prompt, max_length: int = 256):
    model, tokenizer = model_load()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    gen_ids = model.generate(input_ids,
                           max_length=256,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)
    generated = tokenizer.decode(gen_ids[0])
    
    return generated

def test(mesage):
    mesage = mesage['question'] + '\n'

    answer = gpt_chatbot(mesage)

    # return response
    return answer

# mesage =  {
#     "question":"춘곤증인가봐 졸려"
# }

# print(test(mesage))