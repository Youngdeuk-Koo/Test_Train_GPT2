import torch
from transformers import PreTrainedTokenizerFast

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

def model_load():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    device = torch.device("cuda")
    
    print("Current device:", device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained("SAVE_TOKENIZER_DIR")

    model = torch.load("SAVE_MODEL_DIR/model", map_location=device)

    return model, tokenizer

model, koGPT2_TOKENIZER = model_load()

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            prompt = Q_TKN + q + SENT + A_TKN + a
            # print(prompt)
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(prompt)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
            # print('aa', a)
        print("Chatbot > {}".format(a.strip()))