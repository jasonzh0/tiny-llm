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
        self.theta = 1 / (base ** (2 * dim_index / dims))

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        '''
        input size x = N * L * H * D
        cos/sin frequencies = (MAX_SEQ_LEN, D // 2)
        '''
        *N, L, H, D = x.shape

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

        # L * 1 * D // 2  (broadcasts from the right against N.. * L * H * D//2)
        cos_freqs = mx.cos(positions @ thetas)[:, None, :].astype(x.dtype)
        sin_freqs = mx.sin(positions @ thetas)[:, None, :].astype(x.dtype)


        if self.traditional:
            # N * L * H * D//2 * 2
            x = mx.reshape(x, (*N, L, H, D // 2, 2))

            # N * L * H * D//2
            x1 = x[..., 0]
            x2 = x[..., 1]

        else:
            # N * L * H * D//2
            x1 = x[..., :D//2]
            x2 = x[..., D//2:]

        # N * L * H * D//2
        x1_rotated = x1 * cos_freqs - x2 * sin_freqs
        x2_rotated = x1 * sin_freqs + x2 * cos_freqs

        if self.traditional:
            x = mx.stack([x1_rotated, x2_rotated], axis = -1)
        else:
            x = mx.concat([x1_rotated, x2_rotated], axis = -1)

        # N * L * H * D//2 * 2
        # N * L * H * D
        x = mx.reshape(x, (*N, L, H, D))

        return x
