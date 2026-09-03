import mlx.core as mx
from .quantize import QuantizedWeights


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        '''
        weight: vocab_size * embedding_dim
        input: N...
        output: N... * embedding_dim

        index look up on weight from token id x
        equivalent to multiplying one-hot by weight
        '''
        return self.weight[x]



    def as_linear(self, x: mx.array) -> mx.array:
        '''
        weight: vocab_size * embedding_dim
        input: N.. * embedding_dim
        output: N.. * vocab_size
        '''
        weight_t = mx.swapaxes(self.weight, -1, -2)
        return x @ weight_t



class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
        use_custom_kernel: bool = False,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass

    def as_linear(self, x: mx.array) -> mx.array:
        pass
