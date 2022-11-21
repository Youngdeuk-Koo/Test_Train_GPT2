import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
device = torch.device("cuda")
print("Current device:", device)

def model_load():

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
                                                                   bos_token=BOS, 
                                                                   eos_token=EOS, 
                                                                   unk_token="<unk>", 
                                                                   pad_token=PAD, 
                                                                   mask_token=MASK,
                                                                   )

    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

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

def test(q):
    q = q["question"]
    a = ""
    prompt = Q_TKN + q + SENT + A_TKN + a
    answer = gpt_chatbot(prompt)
    return answer
