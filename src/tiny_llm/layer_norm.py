import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        assert weight.shape == (dim,), f"expected weight of shape ({dim},), got {weight.shape}"

        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        '''
        Since it's a type of layer norm we want to normalize across D not N (batch)
        '''
        # N..., D
        x_f32 = x.astype(mx.float32)

        # Broadcast weight from D to N..., D
        rms = mx.rsqrt(mx.mean(mx.square(x_f32), axis = -1) + self.eps)[..., None]
        return (x_f32 * rms * self.weight).astype(x.dtype)
