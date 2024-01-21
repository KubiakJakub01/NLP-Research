'''Module with functions for decoding the output of a neural network.'''
import torch


def gready_decode(model, src, src_mask, max_len, start_symbol, device):
    '''Gready decoding.'''
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys,
            torch.tensor([i + 1]).type(torch.long).to(device)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
