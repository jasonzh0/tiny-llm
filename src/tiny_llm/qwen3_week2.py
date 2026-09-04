from typing import Any

import mlx.core as mx

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding
from .kv_cache import TinyKvCache, TinyKvFullCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    scaled_dot_product_attention,
    swiglu,
)

WEEK2_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "rmsnorm",
    "rope",
    "swiglu",
    "decode-attention",
    "simd-matmul",
    "split-k",
)

DECODE_ATTENTION_MAX_CONTEXT = 256
DECODE_ATTENTION_MAX_QUERY = 2


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_decode_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_decode_attention = use_decode_attention

        self.scale = head_dim**-0.5

        # Week 1 operators, hoisted out of __call__ so they are built once.
        self.q_norm_op = RMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm_op = RMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = RoPE(head_dim, max_seq_len, theta, False)

        # TODO (Day 2+): build the FastRMSNorm / FastRoPE variants here and
        # select between them with use_fast_rms_norm / use_fast_rope.

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape

        # B * L * (H_q * D)
        q = linear(x, self.wq)

        # B * L * H_q * D
        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))

        k = linear(x, self.wk)
        v = linear(x, self.wv)

        # B * L * H * D
        v = mx.reshape(v, (B, L, self.num_kv_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_kv_heads, self.head_dim))

        q = self.q_norm_op(q)
        k = self.k_norm_op(k)

        # TODO (Day 1): rotate at the absolute position of the incoming tokens,
        # i.e. slice(offsets, offsets + L) instead of slice(0, L).
        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        # B * H_q * L * D
        q = mx.swapaxes(q, -2, -3)

        # B * H * L * D
        k = mx.swapaxes(k, -2, -3)
        v = mx.swapaxes(v, -2, -3)

        # TODO (Day 1): append the new k/v to the cache and fetch the full
        # prefix back, e.g.
        #   k, v, offsets, mask = cache.update_and_fetch(k, v, None, mask)
        # after which k/v are B * H * S * D while q stays B * H_q * L * D.

        # query.shape = B * H_q * L * D
        # key.shape = B * H * S * D
        # B * H_q * L * D
        x = scaled_dot_product_attention_grouped(q, k, v, self.scale, mask)

        # B * L * H_q * D
        x = mx.swapaxes(x, -2, -3)

        # B * L * (H_q * D)
        x = mx.reshape(x, (B, L, (self.num_heads * self.head_dim)))

        # B * L * E
        x = linear(x, self.wo)

        return x


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        '''
        E is the hidden size
        I is the intermediate dimension size
        L is the sequence length

        x:  N.. * L * E
        w_gate: I * E
        w_up: I * E
        w_down: E * I
        output: N.. * L * E
        '''

        # N.. * L * I
        gate = silu(x @ mx.swapaxes(self.w_gate, -1, -2))

        # N.. * L * I
        value = x @ mx.swapaxes(self.w_up, -1, -2)

        # TODO (Day 5): route through swiglu() when use_fast_swiglu is set.
        # N.. * L * I
        activation = mx.multiply(gate, value)

        # N.. * L * E
        down = activation @ mx.swapaxes(self.w_down, -1, -2)

        return down


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_decode_attention: bool = True,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.use_fast_rms_norm = use_fast_rms_norm

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, rms_norm_eps
        )
        self.attention = Qwen3MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            max_seq_len,
            theta,
            rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_decode_attention=use_decode_attention,
        )
        self.mlp = Qwen3MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
            use_fast_swiglu=use_fast_swiglu,
        )

        # TODO (Day 4): build the FastRMSNorm variants and select on
        # use_fast_rms_norm.

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x1 = self.input_layernorm(x)
        x1 = self.attention(x1, offset, cache, mask)
        x2 = x1 + x

        x3 = self.post_attention_layernorm(x2)
        x3 = self.mlp(x3)
        x4 = x2 + x3
        return x4


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "split-k",
        use_mlx_quantized_linear: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.mlx_model = mlx_model
        self.checkpoint = checkpoint
        self.use_mlx_quantized_linear = use_mlx_quantized_linear

        # TODO (Day 1): derive the per-kernel flags from `checkpoint` using the
        # cumulative order in WEEK2_CHECKPOINTS. At "kv-cache" every fast path
        # is still off, so the model runs the Week 1 operators.
        use_fast_rms_norm = False
        use_fast_rope = False
        use_fast_swiglu = False
        use_decode_attention = False

        # TODO (Day 3): swap dequantize_linear for QuantizedWeights.from_mlx_layer
        # once the packed-weight kernels exist.
        self.embedding = Embedding(
            mlx_model.args.vocab_size,
            mlx_model.args.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens).astype(mx.bfloat16),
        )
        self.layers = [
            Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=dequantize_linear(layer.self_attn.q_proj).astype(mx.bfloat16),
                wk=dequantize_linear(layer.self_attn.k_proj).astype(mx.bfloat16),
                wv=dequantize_linear(layer.self_attn.v_proj).astype(mx.bfloat16),
                wo=dequantize_linear(layer.self_attn.o_proj).astype(mx.bfloat16),
                q_norm=layer.self_attn.q_norm.weight.astype(mx.bfloat16),
                k_norm=layer.self_attn.k_norm.weight.astype(mx.bfloat16),
                w_gate=dequantize_linear(layer.mlp.gate_proj).astype(mx.bfloat16),
                w_up=dequantize_linear(layer.mlp.up_proj).astype(mx.bfloat16),
                w_down=dequantize_linear(layer.mlp.down_proj).astype(mx.bfloat16),
                w_input_layernorm=layer.input_layernorm.weight.astype(mx.bfloat16),
                w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(
                    mx.bfloat16
                ),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
                use_fast_rms_norm=use_fast_rms_norm,
                use_fast_rope=use_fast_rope,
                use_fast_swiglu=use_fast_swiglu,
                use_decode_attention=use_decode_attention,
            )
            for layer in mlx_model.layers
        ]
        self.rms_norm = RMSNorm(
            mlx_model.args.hidden_size,
            mlx_model.model.norm.weight,
            mlx_model.args.rms_norm_eps,
        )

    def create_kv_cache(self) -> list[TinyKvCache]:
        # TODO (Day 1, Task 3): one request-scoped cache handle per layer.
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        # N * hidden_size
        x = self.embedding(inputs)

        # TODO (Day 1, Task 2): pass each layer its own cache handle plus the
        # caller's offset.
        # N * hidden_size
        for i in range(len(self.layers)):
            x = self.layers[i](x, 'causal')

        # TODO (Day 1): honour logits_to_keep so decode only projects the last
        # row through the vocab-sized lm_head.

        # N * hidden_size
        x = self.rms_norm(x)

        # N * vocab_size
        if self.mlx_model.args.tie_word_embeddings:
            x = self.embedding.as_linear(x)
        else:
            x = self.mlx_model.model.lm_head(x)

        return x
