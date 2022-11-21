import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
device = torch.device("cuda")
print(device)

# tokenizer = AutoTokenizer.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
# )
# model = AutoModelForCausalLM.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   pad_token_id=tokenizer.eos_token_id,
#   torch_dtype='auto', low_cpu_mem_usage=True
# ).to(device='cuda', non_blocking=True)


# tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 

# model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

tokenizer = PreTrainedTokenizerFast.from_pretrained("SAVE_TOKENIZER_DIR", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 
model = GPT2LMHeadModel.from_pretrained('./SAVE_MODEL_DIR/model')

# PATH = './SAVE_MODEL_DIR/'
# torch.save(model, PATH + "model")

# tokenizer.save_pretrained('SAVE_TOKENIZER_DIR')


_ = model.eval()


text = '근육이 커지기 위해서는'
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids]),
                           max_length=128,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)
generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)
