import torch

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)