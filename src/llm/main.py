
import torch
import tiktoken

from .model import GPTBlock, GPT_CONFIG_124_M
from .generate import generate


def main():
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTBlock(GPT_CONFIG_124_M)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124_M["context_length"],
    )

    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
