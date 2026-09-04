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

        self.q_norm_op = RMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm_op = RMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = RoPE(head_dim, max_seq_len, theta, False)

    def __call__(
        self,
        x: mx.array,
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

        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

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
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

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
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.self_attn = Qwen3MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads, head_dim, wq, wk, wv, wo, q_norm, k_norm, max_seq_len, theta, rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x1 = self.input_layernorm(x)
        x1 = self.self_attn(x1, mask)
        x2 = x1 + x

        x3 = self.post_attention_layernorm(x2)
        x3 = self.mlp(x3)
        x4 = x2 + x3
        return x4


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.mlx_model = mlx_model
        self.embedding = Embedding(mlx_model.args.vocab_size, mlx_model.args.hidden_size, dequantize_linear(mlx_model.model.embed_tokens).astype(mx.bfloat16))
        self.layers_inner = [
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
                w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(mx.bfloat16),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            for layer in mlx_model.model.layers
        ]
        self.rms_norm = RMSNorm(mlx_model.args.hidden_size, mlx_model.model.norm.weight, mlx_model.args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # N * hidden_size
        x = self.embedding(inputs)

        # N * hidden_size
        for i in range(len(self.layers_inner)):
            x = self.layers_inner[i](x, 'causal')

        # N * hidden_size
        x = self.rms_norm(x)

        # N * vocab_size
        if self.mlx_model.args.tie_word_embeddings:
            x = self.embedding.as_linear(x)
        else:
            x = self.mlx_model.model.lm_head(x)

        return x
