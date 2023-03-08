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
    N = sum(seq_lens)
    sampler = MultiSeqIndexSampler(seq_lens, N * 2, MockRng())

    sample_idxs_expected = [
        (0, (0, 0)),
        (11, (1, 1)),
        (11 + N, (1, 1)),
        (N * 2 - 1, (4, 3)),
    ]
    for idx, expected in sample_idxs_expected:
        assert sampler[idx] == expected

    all_samples = list(sampler)
    assert len(all_samples) == N * 2
    assert all_samples[0] == (0, 0)
    assert all_samples[1] == (0, 1)
    assert all_samples[N - 1] == (4, 3)
    assert all_samples[N] == (0, 0)
    assert all_samples[N * 2 - 1] == (4, 3)


def test_multi_seq_index_sampler_offset(seq_lens: Sequence[int]):
    N = sum(seq_lens)
    start_offset = 2
    end_offset = 1
    sampler = MultiSeqIndexSampler(seq_lens, N * 2, MockRng(), start_offset=start_offset, end_offset=end_offset)

    sample_idxs_expected = [
        (0, (0, 2)),
        (8, (1, 3)),
        (9, (4, 2)),
        (10, (0, 2)),
    ]
    for idx, expected in sample_idxs_expected:
        assert sampler[idx] == expected

    all_samples = list(sampler)
    assert len(all_samples) == N * 2
    seq_idxs, idxs_seq = zip(*all_samples)
    assert all([s in {0, 1, 4} for s in seq_idxs])
    assert all([start_offset <= i < (sampler._indexer._lengths[s] - end_offset) for s, i in zip(seq_idxs, idxs_seq)])
