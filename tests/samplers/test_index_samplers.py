from math import ceil
from typing import List, Sequence

import pytest

from monopine.samplers.index_samplers import MultiSeqIndexer, MultiSeqIndexSampler


@pytest.fixture
def seq_lens() -> List[int]:
    return [10, 5, 1, 0, 4]


def test_multi_seq_indexer(seq_lens: Sequence[int]):
    indexer = MultiSeqIndexer(seq_lens)
    assert len(indexer) == sum(seq_lens)

    concat_indiv_idx_pairs = [
        (0, (0, 0)),
        (len(indexer) - 1, (len(seq_lens) - 1, seq_lens[-1] - 1)),
        (seq_lens[0] + 2, (1, 2))
    ]

    for cat_idx, indiv_idx in concat_indiv_idx_pairs:
        assert indexer[cat_idx] == indiv_idx # Test indexing into concatenated sequences
        assert indexer[indiv_idx] == cat_idx # Test indexing into individual sequences
    with pytest.raises(IndexError):
        indexer[len(indexer)]
    with pytest.raises(IndexError):
        indexer[(len(seq_lens), 0)]
    with pytest.raises(IndexError):
        indexer[(0, seq_lens[0])]


class MockRng:
    def __init__(self):
        self._counter = 0
    
    def randrange(self, range_: int):
        out = self._counter % range_
        self._counter += 1
        return out


def test_multi_seq_index_sampler(seq_lens: Sequence[int]):
    L = sum(seq_lens)
    N = 2 * L
    sampler = MultiSeqIndexSampler(seq_lens, N, MockRng())

    sample_idxs_expected = [
        (0, (0, 0)),
        (11, (1, 1)),
        (11 + L, (1, 1)),
        (N - 1, (4, 3)),
    ]
    for idx, expected in sample_idxs_expected:
        assert sampler[idx] == expected

    all_samples = list(sampler)
    assert len(all_samples) == N
    assert all_samples[0] == (0, 0)
    assert all_samples[1] == (0, 1)
    assert all_samples[L - 1] == (4, 3)
    assert all_samples[L] == (0, 0)
    assert all_samples[N - 1] == (4, 3)


def test_multi_seq_index_sampler_offset(seq_lens: Sequence[int]):
    L = sum(seq_lens)
    N = 2 * L
    start_offset = 2
    end_offset = 1
    sampler = MultiSeqIndexSampler(seq_lens, N, MockRng(), start_offset=start_offset, end_offset=end_offset)

    sample_idxs_expected = [
        (0, (0, 2)),
        (8, (1, 3)),
        (9, (4, 2)),
        (10, (0, 2)),
    ]
    for idx, expected in sample_idxs_expected:
        assert sampler[idx] == expected

    all_samples = list(sampler)
    assert len(sampler) == len(all_samples) == N
    seq_idxs, idxs_seq = zip(*all_samples)
    assert all([s in {0, 1, 4} for s in seq_idxs])
    assert all([start_offset <= i < (sampler._indexer._lengths[s] - end_offset) for s, i in zip(seq_idxs, idxs_seq)])

    # Make sure invalid sampler cannot be created
    with pytest.raises(ValueError):
        max_len = max(seq_lens)
        offset = ceil(max_len / 2)
        invalid_sampler = MultiSeqIndexSampler(seq_lens, N, MockRng(), start_offset=offset, end_offset=offset)


def test_multi_seq_index_sampler_all(seq_lens: Sequence[int]):
    L_all = sum(seq_lens)
    sampler_all = MultiSeqIndexSampler(seq_lens, "all", MockRng())
    assert len(sampler_all) == L_all
    all_samples_all = list(sampler_all)
    assert len(all_samples_all) == L_all
    assert all([
        all_samples_all[0] == (0, 0),
        all_samples_all[1] == (0, 1),
        all_samples_all[seq_lens[0]] == (1, 0),
        all_samples_all[-1] == (len(seq_lens) - 1, seq_lens[-1] - 1),
    ])

    start_offset = 2
    end_offset = 1
    sampler_offsets = MultiSeqIndexSampler(seq_lens, "all", MockRng(), start_offset=start_offset, end_offset=end_offset)
    L_offsets = sum([max(l - end_offset - start_offset, 0) for l in seq_lens])
    all_samples_offsets = list(sampler_offsets)
    assert len(all_samples_offsets) == L_offsets
    assert all([
        all_samples_offsets[0] == (0, start_offset),
        all_samples_offsets[-1] == (len(seq_lens) - 1, seq_lens[-1] - end_offset - 1),
    ])
