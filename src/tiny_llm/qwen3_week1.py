import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
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

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape

        # B * L * E
        q = linear(x, self.wq)

        # B * L * H_q * D
        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))

        k = linear(x, self.wk)
        v = linear(x, self.wv)

        # B * L * H * D
        v = mx.reshape(v, (B, L, self.num_kv_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_kv_heads, self.head_dim))

        q = mx.fast.rms_norm(q, self.q_norm, self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, self.rms_norm_eps)

        q = RoPE(self.head_dim, L, self.theta, False)(q, offset=slice(0, L))
        k = RoPE(self.head_dim, L, self.theta, False)(k, offset=slice(0, L))

        # B * H_q * L * D
        q = mx.swapaxes(q, -2, -3)

        # B * H * L * D
        k = mx.swapaxes(k, -2, -3)
        v = mx.swapaxes(v, -2, -3)

        # query.shape = B * H_q * L * D
        # key.shape = B * H * S * D
        # B * H_q * L * D
        x = scaled_dot_product_attention_grouped(q, k, v, None, mask)

        # B * L * H_q * D
        x = mx.swapaxes(x, -2, -3)

        # B * L * E
        x = mx.reshape(x, (B, L, self.hidden_size))

        # B * L * E
        x = linear(x, self.wo)

        return x



class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
