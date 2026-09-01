import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    '''
    input tensor shape: N.. * I
    weight tensor shape: O * I
    biases tensor shape: O * I
    output tensor shape: N.. * O
    '''
    value = x @ w.T
    if bias is not None:
        value = value + bias
    return value


def silu(x: mx.array) -> mx.array:
    pass
