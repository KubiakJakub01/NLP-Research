"""Module with functions for decoding the output of a neural network."""
import torch
from transformers import top_k_top_p_filtering


def gready_decode(model, src, src_mask, max_len, start_symbol, device):
    """Gready decoding."""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, torch.tensor([i + 1]).type(torch.long).to(device))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, device, beam_size=5):
    """Beam search decoding."""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    beam = [(ys, 0)]
    for i in range(max_len - 1):
        candidates = []
        for snt, score in beam:
            out = model.decode(
                memory, src_mask, snt, torch.tensor([i + 1]).type(torch.long).to(device)
            )
            prob = model.generator(out[:, -1])
            topv, topi = prob.topk(beam_size)
            for j in range(beam_size):
                candidates.append(
                    (torch.cat([snt, topi[:, j].unsqueeze(1)], dim=1), score + topv[:, j].item())
                )
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    return beam[0][0]


def top_k_top_p_decode(model, src, src_mask, max_len, start_symbol, device, top_k=0, top_p=0.0):
    """Top-k top-p decoding."""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, torch.tensor([i + 1]).type(torch.long).to(device))
        prob = model.generator(out[:, -1])
        filtered_logits = top_k_top_p_filtering(prob, top_k=top_k, top_p=top_p)
        next_word = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def multinomial_decode(model, src, src_mask, max_len, start_symbol, device):
    """Multinomial decoding."""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, torch.tensor([i + 1]).type(torch.long).to(device))
        prob = model.generator(out[:, -1])
        next_word = torch.multinomial(torch.softmax(prob, dim=-1), num_samples=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
