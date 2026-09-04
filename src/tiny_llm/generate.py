import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> None:
    def _step(model, y):
        # 1, S
        y = y[None, :]

        # 1, S, vocab_size
        y = model(y)

        # (1, vocab_size)
        logits = y[:, -1, :]

        # (1, vocab_size)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        if sampler is None:
            return mx.argmax(log_probs, axis = -1)
        else:
            return sampler(log_probs)
    
    x = mx.array(tokenizer.encode(prompt, add_special_tokens=False))

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    while len(detokenizer.tokens) < 256:
        next_token = _step(model, x)
        x = mx.concat([x, next_token])
        detokenizer.add_token(next_token.item())
        
        if detokenizer.tokens[-1] == tokenizer.eos_token_id:
            break

        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()
    print()


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        # B, L
        y = y[None, :]

        # B, logits_to_keep, vocab_size
        y = model(y, offset, kv_cache, 1)

        # B, vocab_size
        logits = y[:, -1, :]

        # B, vocab_size
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # (B,)
        return mx.argmax(log_probs, axis = -1)

    kv_cache = model.create_kv_cache()
    x = mx.array(tokenizer.encode(prompt, add_special_tokens=False))

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    offset = 0

    while len(detokenizer.tokens) < 256:
        next_token = _step(model, x, offset, kv_cache)
        offset += len(x)

        detokenizer.add_token(next_token.item())
        if detokenizer.tokens[-1] == tokenizer.eos_token_id:
            break

        print(detokenizer.last_segment, end="", flush=True)

        x = next_token

    detokenizer.finalize()
    print()

def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
) -> str:
    pass
