import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: mx.array):
        '''
        B = batch size
        V = vocab size
        logprobs: B * V
        '''
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        #   probs    = [0.05,  0.40,  0.10,  0.30,  0.15]
        #   logprobs = [-3.0, -0.92, -2.3,  -1.2,  -1.9]

        logprobs = logprobs / temp

        if top_k is not None:

            # [0, 2, 4, 3, 1]
            top_k_index = mx.argpartition(logprobs, top_k)

            # top_k_index[:, :-2] = [0, 2, 4]
            # [-inf, -0.92, -inf, -1.2, -inf]
            logprobs = mx.put_along_axis(logprobs, top_k_index[:, :-top_k], mx.array(-mx.inf), axis=-1)
            
        if top_p is not None:  
            logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)

            # [1, 3, 4, 2, 0]
            order = mx.argsort(-logprobs, axis=-1)

            # [-0.92, -1.2, -1.9, -2.3, -3.0]
            # Gather: take from index order[i] and place at position i
            logprobs_sorted = mx.take_along_axis(logprobs, order, axis=-1)

            # [0.40, 0.30, 0.15, 0.10, 0.05]
            probs_sorted = mx.exp(logprobs_sorted)

            # [0.40, 0.70, 0.85, 0.95, 1.00]
            probs_cum = mx.cumsum(probs_sorted, axis=-1)
            
            # [0, 0, 0, 1, 1] for top_p = 0.8
            mask_sorted = (probs_cum - probs_sorted) >= top_p

            # [1, 0, 1, 0, 0]
            # Scatter: take position i and place at order[i]
            mask = mx.put_along_axis(mx.zeros_like(mask_sorted), order, mask_sorted, axis=-1)

            # [-inf, -0.92, -inf, -1.2, -1.9]
            logprobs = mx.where(mask, -mx.inf, logprobs)

        return mx.random.categorical(logprobs)

    return sample
