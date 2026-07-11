#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batching strategies for streaming/iterable datasets.

Provides length-based grouping and packing for efficient training with
variable-length audio.

Key classes:
- ``PackingIterableDataset``: Packs multiple samples into fixed-length sequences
  for training. Used by ``omnivoice.training.builder`` with flex_attention.
- ``StreamLengthGroupDataset``: Groups samples by length into buckets. Used by
  data processing scripts (e.g. ``omnivoice/scripts/``) and by
  ``omnivoice.training.builder`` when ``attn_implementation != "flex_attention"``.
"""

import bisect
import logging

logger = logging.getLogger(__name__)
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from omnivoice.data.dataset import IterableDataReader, WrappedIterableDataset


class StreamLengthGroupDataset(WrappedIterableDataset):
    """A streaming dataset that groups samples by their lengths into buckets.

    By default, length is measured as audio duration in seconds from a raw
    waveform field. Pass a custom ``length_fn`` to use a different measure —
    e.g. ``lambda s: s["length"]`` for processed training data, in which case
    ``batch_duration`` and ``min/max_length`` should use the same units.

    If ``processor`` is provided, each raw sample is processed before length
    measurement and bucketing, and the yielded batches contain **processed**
    samples. This allows accurate bucketing by post-processing token length
    (used in the SDPA training path).
    """

    def __init__(
        self,
        dataset: IterableDataReader,
        batch_duration: float,
        min_length: float = 0.5,
        max_length: float = 30.0,
        num_buckets: int = 20,
        audio_key: str = "audio",
        drop_last: bool = False,
        max_sample: Optional[int] = None,
        length_fn: Optional[Any] = None,
        processor: Optional[Any] = None,
    ):
        self.dataset = dataset
        self.batch_duration = batch_duration
        self.min_length = min_length
        self.max_length = max_length
        self.num_buckets = num_buckets
        self.audio_key = audio_key
        self.drop_last = drop_last
        self.max_sample = max_sample if max_sample is not None else float("inf")
        self.length_fn = length_fn
        self.processor = processor

        self.boundaries = np.linspace(min_length, max_length, num_buckets + 1)[1:]

    def set_epoch(self, epoch: int):
        """
        Set the epoch for shuffling.
        """
        self.dataset.set_epoch(epoch)

    def _get_bucket_id(self, length: float) -> int:

        return bisect.bisect_left(self.boundaries, length)

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        buckets = [[] for _ in range(self.num_buckets)]
        bucket_max_len = [0.0] * self.num_buckets

        for sample in self.dataset:
            if self.processor is not None:
                try:
                    sample = self.processor(sample)
                except Exception as e:
                    logging.warning(f"Error processing sample: {e}")
                    continue

            if self.length_fn is not None:
                duration = self.length_fn(sample)
            else:
                audio = sample[self.audio_key]
                duration = audio.size(-1) / self.dataset.sample_rate

            if duration < self.min_length or duration > self.max_length:
                # logging.warning(f"Skipping sample with duration {duration:.2f}s")
                continue

            b_id = self._get_bucket_id(duration)
            buckets[b_id].append(sample)

            if duration > bucket_max_len[b_id]:
                bucket_max_len[b_id] = duration

            if (
                bucket_max_len[b_id] * (len(buckets[b_id]) + 1) >= self.batch_duration
                or len(buckets[b_id]) >= self.max_sample
            ):
                yield buckets[b_id]
                buckets[b_id] = []
                bucket_max_len[b_id] = 0.0

        if not self.drop_last:
            for b_idx, bucket in enumerate(buckets):
                if bucket:
                    yield bucket
                    buckets[b_idx] = []


class PackingIterableDataset(WrappedIterableDataset):
    """
    An IterableDataset that dynamically processes samples using a processor
    and packs them into batches based on the real token count.

    Args:
        dataset (Iterable): The raw dataset to process.
        processor (Callable): A processor to process each sample.
        batch_tokens (int): Maximum number of tokens per batch.
    """

    def __init__(
        self,
        dataset: IterableDataReader,
        processor: Any,
        batch_tokens: int,
        balanced_window: int = 0,
    ):
        self.dataset = dataset
        self.processor = processor
        self.batch_tokens = batch_tokens
        self.skip_batches = 0
        self.balanced_window = balanced_window

    def set_epoch(self, epoch: int):
        """
        Set the epoch for shuffling.
        """
        self.dataset.set_epoch(epoch)

    @staticmethod
    def _attn_cost(sample):
        """Analytic visible-pair proxy for the B2 dual-copy mask: p^2 + 2ph +
        h^2 + 32h (prefix/clean/noisy visibility). Falls back to length^2."""
        tag = sample.get("copy_tag")
        if tag is None:
            n = sample["length"]
            return float(n) * n
        p = float((tag == 0).sum())
        h = float((tag == 1).sum())
        return p * p + 2.0 * p * h + h * h + 32.0 * h

    def _iter_balanced(self):
        """Windowed cost-balanced packing: buffer ~window packs of samples,
        LPT-assign (largest cost first -> currently cheapest pack with room).
        Pure sample->pack reassignment; every sample still trains exactly
        once, so this is equivalent to a shuffle-seed change, not a recipe
        change. Leftovers carry into the next window."""
        window_tokens = self.balanced_window * self.batch_tokens
        buf = []
        buf_tokens = 0

        def flush(buf, final=False):
            buf.sort(key=lambda x: -x[0])
            packs = []   # list of [cost_sum, token_sum, samples]
            for cost, sample in buf:
                n = sample["length"]
                candidates = [pk for pk in packs if pk[1] + n <= self.batch_tokens]
                if candidates:
                    pk = min(candidates, key=lambda x: x[0])
                    pk[0] += cost; pk[1] += n; pk[2].append(sample)
                else:
                    packs.append([cost, n, [sample]])
            packs.sort(key=lambda x: -x[1])
            carry = []
            if not final and packs:
                # retain the least-filled pack as carry-over seed
                last = packs.pop() if len(packs) > 1 else None
                if last is not None:
                    carry = [(self._attn_cost(sm), sm) for sm in last[2]]
            for pk in packs:
                yield pk[2]
            self._carry = carry

        self._carry = []
        for raw_sample in self.dataset:
            try:
                processed = self.processor(raw_sample)
            except Exception as e:
                logging.warning(f"Error processing sample {raw_sample}: {e}")
                continue
            n = processed["length"]
            if n > self.batch_tokens:
                continue
            buf.append((self._attn_cost(processed), processed))
            buf_tokens += n
            if buf_tokens >= window_tokens:
                yield from flush(buf)
                buf = list(self._carry); self._carry = []
                buf_tokens = sum(x[1]["length"] for x in buf)
        if buf:
            yield from flush(buf, final=True)

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        if self.balanced_window and self.balanced_window > 0:
            yield from self._iter_balanced()
            return
        current_batch = []
        current_token_count = 0

        for raw_sample in self.dataset:
            # Process the sample using the processor
            try:
                processed_sample = self.processor(raw_sample)
            except Exception as e:
                logging.warning(f"Error processing sample {raw_sample}: {e}")
                continue

            sample_length = processed_sample["length"]

            if sample_length > self.batch_tokens:
                self.dropped_overlong = getattr(self, "dropped_overlong", 0) + 1
                if self.dropped_overlong % 100 == 1:
                    logger.warning(
                        f"packer dropped overlong sample #{self.dropped_overlong} "
                        f"(length={sample_length} > batch_tokens={self.batch_tokens})"
                    )
                continue

            # Check if adding this sample exceeds the batch token limit
            if current_token_count + sample_length > self.batch_tokens:
                # Yield the current batch and start a new one
                yield current_batch
                current_batch = []
                current_token_count = 0

            # Add the processed sample to the current batch
            current_batch.append(processed_sample)
            current_token_count += sample_length

        # Yield the last batch if it's not empty
        if current_batch:
            yield current_batch
