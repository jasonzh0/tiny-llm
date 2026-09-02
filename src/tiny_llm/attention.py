import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    '''
    tensor shape: N.. * L * D
    N = batch
    L = sequence length
    D = head dimensions
    '''
    d = query.shape[-1] # attention head dimensions
    if scale is None:
        scale = 1/mx.sqrt(d)

    attend_score = (query @ mx.swapaxes(key, -2, -1)) * scale

    if mask is not None:
        attend_score = attend_score + mask
    
    attention = softmax(attend_score, axis = -1) @ value
    return attention


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        '''
        input tensor shape = N * L * E = N * L * H * D
        N = batch
        H = # of attention heads
        L = sequence length
        D = head dimension

        E = H * D = embed dimensions
        '''
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dimensions = hidden_size // num_heads
        self.wq = wq # E * E = (H * D) * E
        self.wk = wk
        self.wv = wv
        self.wo = wo # E * E = E * (H X D)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # N * L * E
        *N, L, E = query.shape

        query_result = query @ self.wq.T
        key_result = key @ self.wk.T
        value_result = value @ self.wv.T

        # note: more intuitively we could reshape q,k,v into shape N * H * D * E and then N * H * E * D
        # as the different heads and then we use to multiply to result in N * H * L * D

        # N * L * H * D
        query_result = mx.reshape(query_result, (*N, L, self.num_heads, self.head_dimensions))
        key_result = mx.reshape(key_result, (*N, L, self.num_heads, self.head_dimensions))
        value_result = mx.reshape(value_result, (*N, L, self.num_heads, self.head_dimensions))

        # N * H * L * D
        query_t = mx.swapaxes(query_result, -3, -2)
        key_t = mx.swapaxes(key_result, -3, -2)
        value_t = mx.swapaxes(value_result, -3, -2)

        # N * H * L * D
        attention = scaled_dot_product_attention_simple(query_t, key_t, value_t, None, mask)

        # N * L * H * D
        attention_t = mx.swapaxes(attention, -3, -2)

        # N * L * E
        attention_t = mx.reshape(attention_t, (*N, L, E))

        # N * L * E
        return linear(attention_t, self.wo)




def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
