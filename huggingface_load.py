from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"

class load():
    def tokenizer_load():
        koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
                                                                   bos_token=BOS, 
                                                                   eos_token=EOS, 
                                                                   unk_token="<unk>", 
                                                                   pad_token=PAD, 
                                                                   mask_token=MASK,
                                                                   )
        koGPT2_TOKENIZER.save_pretrained('SAVE_TOKENIZER_DIR')
        
        koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("SAVE_TOKENIZER_DIR")
        
        return koGPT2_TOKENIZER
    
    def model_load():
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        
        return model