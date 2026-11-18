"""
Gemma-2 caption embeddings for training, encoded exactly as SanaPipeline encodes them at inference.

Positive prompts get SanaPipeline's default complex human instruction (CHI) prefix and the
"BOS + last 299 tokens" selection; the null (dropped-caption) prompt is "" without CHI, the same as
SanaPipeline's negative prompt. Embeddings are stored trimmed to their unmasked tokens: SANA's cross-attention
has no positional encoding on keys and masked keys get a −10000 bias, so dropping them is exact.
"""

import inspect

import torch
from diffusers import SanaPipeline

SEQUENCE_LENGTH = 300  # SanaPipeline max_sequence_length
CHI = inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default


@torch.no_grad()
def encode_captions(pipe, captions, batch_size=128, null=False):
    """
    Query (runs the text encoder on the GPU). Encodes captions like SanaPipeline and trims each to its unmasked tokens.

    Args:
        pipe (SanaPipeline): pipeline holding tokenizer + text_encoder on the target device
        captions (list[str]): N captions
        batch_size (int): captions per text-encoder call
        null (bool): encode as the negative/null prompt (no CHI), as SanaPipeline does for ""

    Returns:
        list[torch.Tensor]: N tensors, each (L_i, 2304) bf16 on CPU, L_i = number of unmasked tokens
    """
    out = []
    for i in range(0, len(captions), batch_size):
        chunk = captions[i : i + batch_size]
        if null:
            emb, mask = pipe._get_gemma_prompt_embeds(prompt=chunk, device=pipe.text_encoder.device, dtype=torch.bfloat16,
                                                      max_sequence_length=SEQUENCE_LENGTH, complex_human_instruction=False)
        else:
            emb, mask, _, _ = pipe.encode_prompt(chunk, do_classifier_free_guidance=False, device=pipe.text_encoder.device,
                                                 max_sequence_length=SEQUENCE_LENGTH, complex_human_instruction=CHI)
        out += [e[m.bool()].to(torch.bfloat16).cpu() for e, m in zip(emb, mask)]
    return out


def pad_embeddings(embeddings, length=None):
    """
    Pure function. Right-pads variable-length token embeddings into a batch with an attention mask.

    Args:
        embeddings (list[torch.Tensor]): B tensors, each (L_i, D)
        length (int | None): padded length (≥ every L_i); None = max L_i. A fixed length keeps compiled shapes fixed.

    Returns:
        tuple: (B, max L_i, D) embeddings (zeros where padded), (B, max L_i) int64 mask (1 = real token)

    Examples:
        >>> e, m = pad_embeddings([torch.ones(2, 3), torch.ones(1, 3)])
        >>> tuple(e.shape), m.tolist(), e[1, 1].tolist()
        ((2, 2, 3), [[1, 1], [1, 0]], [0.0, 0.0, 0.0])
        >>> pad_embeddings([torch.ones(2, 3)], length=4)[1].tolist()
        [[1, 1, 0, 0]]
    """
    length = length or max(len(e) for e in embeddings)
    batch = embeddings[0].new_zeros(len(embeddings), length, embeddings[0].shape[1])
    mask = torch.zeros(len(embeddings), length, dtype=torch.int64)
    for i, e in enumerate(embeddings):
        batch[i, : len(e)] = e
        mask[i, : len(e)] = 1
    return batch, mask
