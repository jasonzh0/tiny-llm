import mlx.core as mx

class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        dim_index = mx.arange(dims//2)
        self.theta = 1 / (10000 ** (2 * dim_index / dims))

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        '''
        input size x = N * L * H * D
        cos/sin frequencies = (MAX_SEQ_LEN, D // 2)
        '''
        N, L, H, D = x.shape

        if offset is None:
            start = 0
        elif isinstance(offset, slice):
            start = offset.start
        else:
            pass

        # L x 1
        positions = mx.arange(start, start + L)[:, None]

        # 1 x D // 2
        thetas = self.theta[None, :]

        # 1 * L * 1 * D // 2
        cos_freqs = mx.cos(positions @ thetas)[None, :, None, :]
        sin_freqs = mx.sin(positions @ thetas)[None, :, None, :]

        # N * L * H * D//2 * 2
        x = mx.reshape(x, (N, L, H, D // 2, 2))

        # N * L * H * D//2
        x_evens = x[..., 0]
        x_odds = x[..., 1]

        # N * L * H * D//2
        evens = x_evens * cos_freqs - x_odds * sin_freqs
        odds = x_evens * sin_freqs + x_odds * cos_freqs

        # N * L * H * D//2 * 2
        x = mx.stack([evens, odds], axis = -1)
        # N * L * H * D
        x = mx.reshape(x, (N, L, H, D))

        return x
