from functools import partial
import random
from typing import List, Optional, Sequence, Tuple, Union


class Counter:
    def __init__(self, n: int):
        self._n = n
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self) -> int:
        return self._n

    def __next__(self) -> int:
        i = self._i
        self._i += 1
        if self._i > self._n:
            raise StopIteration
        else:
            return i

    def __call__(self) -> int:
        return next(self)


class MultiSeqIndexer:
    """Helper class for indexing into a concatenated sequence of sequences.

    Initialize with a sequence of sub-sequence lengths. Integer indexing into the concatenated sequences or tuple
    indexing into which sequence and inside the sequence can be used, returning the corresponding integer or tuple
    index, respectively.

    Example:
        For concatenating the following three sequences: ['a', 'b', 'c'], ['x', 'y', 'z', 'AA'], [1.1, None]
        Initialize an indexer: indexer = MultiSeqIndexer([3, 4, 2])
        indexer[5] == (1, 1)
        indexer[(2, 1)] == 8
        indexer[10]  # raises IndexError
    """

    def __init__(self, lengths: Sequence[int]):
        if len(lengths) == 0:
            raise ValueError("At least one length required")
        self._lengths = lengths
        self._starts_ends: List[Tuple[int, int]] = []
        last_end = 0
        for seq_len in lengths:
            self._starts_ends.append((last_end, last_end + seq_len))
            last_end = self._starts_ends[-1][1]

    def __len__(self) -> int:
        return self._starts_ends[-1][1]
    
    def _to_concat_index(self, index: Tuple[int, int]) -> int:
        idx_of_seq, idx_in_seq = index
        if idx_of_seq < 0 or idx_in_seq < 0:
            raise NotImplementedError(f"Negative indexing not supported for type {type(self).__name__}")
        if idx_in_seq >= self._starts_ends[idx_of_seq][1]:
            raise IndexError(f"{type(self).__name__} index out of range")
        return self._starts_ends[idx_of_seq][0] + idx_in_seq
    
    def _to_sub_index(self, index: int) -> Tuple[int, int]:
        if index < 0:
            raise NotImplementedError(f"Negative indexing not supported for type {type(self).__name__}")
        for i, (start, end) in enumerate(self._starts_ends):
            if index < end:
                return i, index - start
        raise IndexError(f"{type(self).__name__} index out of range")
    
    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Union[Tuple[int, int], int]:
        try:
            if int(index) == index:
                return self._to_sub_index(index)
            else:
                raise ValueError("Non-integer index")
        except TypeError:
            if len(index) != 2:
                raise ValueError("Index pair expected: (index_of_sequence, index_in_sequence)")
            return self._to_concat_index(index)
        

class MultiSeqIndexSampler:
    """Random sampler of indices within multiple sequences.

    Utilizes the MultiSeqIndexer, but the returned samples are tuples indexing which sequence and within that sequence.
    Specify the number of samples, as well as offsets from the start and end of each sequence from which the samples may
    be drawn. For example, if start_offset=3 and end_offset=2, and the first of the sequences has length 10, indices
    from this sequence that may be drawn are [2...7].
    """

    def __init__(
            self,
            lengths: Sequence[int],
            num_samples: Optional[int] = None,
            rng=None,
            seed=None,
            start_offset: int = 0,
            end_offset: int = 0,
    ) -> None:
        self._indexer = MultiSeqIndexer(lengths)
        self._total_len = len(self._indexer)
        if num_samples:
            if isinstance(rng, (int, float, str, bytes, bytearray)) and seed is None:
                seed = rng
                rng = None
            if rng is None:
                rng = random.Random(seed)
            self._sampler = partial(rng.randrange, len(self._indexer))
            self._num_samples = int(num_samples)
        else:
            self._num_samples = 0
            for len_ in lengths:
                self._num_samples += max(len_ - end_offset - start_offset, 0)
            self._sampler = Counter(self._total_len)
        self._start_offset = start_offset
        self._end_offset = end_offset
        self._current_idx = 0
        self._samples = None
        self._validate()  # raises ValueError if offsets and sequence lengths not compatible

    def _validate(self):
        for seq_len in self._indexer._lengths:
            if (self._start_offset + self._end_offset) < seq_len:
                return None
        raise ValueError("No valid samples given start and end offset values for the sequence lengths.")

    def __len__(self) -> int:
        return self._num_samples

    def _get_next_sample(self) -> Optional[Tuple[int, int]]:
        next_concat_idx = self._sampler()
        idx_of_seq, idx_in_seq = self._indexer[next_concat_idx]
        if self._start_offset <= idx_in_seq < (self._indexer._lengths[idx_of_seq] - self._end_offset):
            return idx_of_seq, idx_in_seq
        else:
            return None

    def __next__(self) -> Tuple[int, int]:
        if self._current_idx >= self._num_samples:
            raise StopIteration
        idx_tuple = None
        while idx_tuple is None:
            idx_tuple = self._get_next_sample()
        self._current_idx += 1
        return idx_tuple

    def __iter__(self):
        self._current_idx = 0
        return self

    def __getitem__(self, index: int):
        if -len(self) <= index < len(self):
            if self._samples is None:
                self._samples = list(self)
            return self._samples[index]
        else:
            raise IndexError(f"{type(self).__name__} index out of range")
