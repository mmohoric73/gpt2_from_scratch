
import torch


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        idx_next = torch.argmax(
            logits, dim=-1, keepdim=True
        )

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
