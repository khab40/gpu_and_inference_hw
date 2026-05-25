"""
HW3: Mini Inference Engine
CacheManager · Continuous Batching · Prefix Caching

Edit only this file.  See README.md for background and implementation details.

Run:
    python hw3_inference_engine/hw3_task.py
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from engine_utils import (
    CacheHandle,
    Request,
    Batch,
    BatchPhase,
    StepMetrics,
    DummyLLM,
    SchedulingPolicy,
    RequestStatus,
    generate_workload,
    compute_stats,
    print_stats,
    plot_results,
    plot_policy_results,
    BLOCK_SIZE,
    NUM_BLOCKS,
    MAX_SEQS,
    TOKEN_BUDGET,
    PREFILL_CHUNK,
)


# ── Task 1: Cache Manager ─────────────────────────────────────────────────────


class CacheManager:
    """
    Unified block allocator, prefix cache, and LRU eviction.

    Ref-count semantics:
        allocate(n)    ref = 1   request owns the block
        lock(handle)   ref += 1  request also pins a cached block
        unlock(handle) ref -= 1  block is evictable once ref drops to 1
        free(ids)      ref -= 1  block goes to free pool when ref reaches 0
        _evict_blocks_from_kv_cache(n)      reclaims n LRU unlocked blocks from the prefix cache
    """

    def __init__(
        self, num_blocks: int = NUM_BLOCKS, block_size: int = BLOCK_SIZE
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))  # available block IDs
        self._ref: list[int] = [0] * num_blocks  # reference counts
        # Prefix cache: token-tuple key → list of block IDs
        self._cache: dict[tuple[int, ...], list[int]] = {}
        # LRU order: index 0 = least-recently used; updated on every hit and insert
        self._lru: list[tuple[int, ...]] = []
        # Per-block count of how many cache entries reference it.
        # _ref is incremented only ONCE for cache ownership (when _cache_ref
        # goes from 0 → 1) and decremented when _cache_ref returns to 0.
        self._cache_ref: list[int] = [0] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        # TODO
        return len(self._free)

    @property
    def ref_counts(self) -> list[int]:
        """Snapshot of per-block effective ownership refs."""
        return list(self._ref)

    @property
    def cache_ref_counts(self) -> list[int]:
        """Snapshot of per-block cache-entry reference counts."""
        return list(self._cache_ref)

    @property
    def cache_entries(self) -> dict[tuple[int, ...], list[int]]:
        """Snapshot of cached prefix -> block mapping."""
        return {k: list(v) for k, v in self._cache.items()}

    @property
    def lru_keys(self) -> list[tuple[int, ...]]:
        """Snapshot of cache keys in LRU order (oldest first)."""
        return list(self._lru)

    def allocate(self, n: int) -> list[int] | None:
        """Claim n blocks (ref=1 each). Evicts LRU cache entries if needed.
        Returns None only when eviction cannot free enough blocks."""
        # TODO
        if n <= 0:
            return []
        if len(self._free) < n:
            self._evict_blocks_from_kv_cache(n - len(self._free))
        if len(self._free) < n:
            return None

        claimed = self._free[:n]
        del self._free[:n]
        for block_id in claimed:
            self._ref[block_id] = 1
        return claimed

    def free(self, block_ids: list[int]) -> None:
        """Decrement each block's ref; return to the free list when ref reaches 0."""
        # TODO
        for block_id in block_ids:
            self._ref[block_id] -= 1
            if self._ref[block_id] < 0:
                raise RuntimeError(f"negative ref count for block {block_id}")
            if self._ref[block_id] == 0:
                self._free.append(block_id)

    def lock(self, handle: CacheHandle) -> None:
        """Pin the matched blocks (incr ref). Must be called before using them."""
        # TODO
        for block_id in handle.matched_blocks:
            self._ref[block_id] += 1

    def unlock(self, handle: CacheHandle) -> None:
        """Release the pin (decr ref). Blocks become evictable when ref drops to 1."""
        # TODO
        for block_id in handle.matched_blocks:
            self._ref[block_id] -= 1
            if self._ref[block_id] < 0:
                raise RuntimeError(f"negative ref count for block {block_id}")

    def match_prefix(self, tokens: list[int]) -> CacheHandle:
        """Longest-prefix lookup. Returns a CacheHandle WITHOUT pinning.
        Updates LRU order on a hit. Returns CacheHandle(0, []) on a miss."""
        # TODO
        max_complete = len(tokens) // self.block_size
        for n_blocks in range(max_complete, 0, -1):
            n_tokens = n_blocks * self.block_size
            key = tuple(tokens[:n_tokens])
            blocks = self._cache.get(key)
            if blocks is None:
                continue
            self._lru.remove(key)
            self._lru.append(key)
            return CacheHandle(n_tokens, list(blocks))
        return CacheHandle(0, [])

    def insert_prefix(self, tokens: list[int], block_ids: list[int]) -> None:
        """Store every complete-block prefix not already cached.
        For each block in a new entry, increment _cache_ref. Only increment
        _ref when _cache_ref goes from 0 → 1 (first cache entry for that block)
        so that overlapping entries share a single ref-count for cache ownership."""
        # TODO
        complete_blocks = min(len(tokens) // self.block_size, len(block_ids))
        for n_blocks in range(1, complete_blocks + 1):
            n_tokens = n_blocks * self.block_size
            key = tuple(tokens[:n_tokens])
            if key in self._cache:
                continue

            blocks = list(block_ids[:n_blocks])
            self._cache[key] = blocks
            self._lru.append(key)
            for block_id in blocks:
                if self._cache_ref[block_id] == 0:
                    self._ref[block_id] += 1
                self._cache_ref[block_id] += 1

    def _evict_blocks_from_kv_cache(self, n: int) -> None:
        """Attempt to evict least-recently-used cache entries whose blocks are
        unlocked (`ref == 1`) to reclaim up to `n` blocks.
        Because cache entries can overlap on blocks, evicting an entry does not
        always free a block immediately. A block becomes free only when its
        cache ownership drops to zero."""
        # TODO
        if n <= 0:
            return

        freed = 0
        idx = 0
        while freed < n and idx < len(self._lru):
            key = self._lru[idx]
            blocks = self._cache[key]
            if not all(self._ref[block_id] == 1 for block_id in blocks):
                idx += 1
                continue

            self._lru.pop(idx)
            del self._cache[key]
            for block_id in blocks:
                self._cache_ref[block_id] -= 1
                if self._cache_ref[block_id] < 0:
                    raise RuntimeError(f"negative cache ref count for block {block_id}")
                if self._cache_ref[block_id] == 0:
                    self._ref[block_id] -= 1
                    if self._ref[block_id] < 0:
                        raise RuntimeError(f"negative ref count for block {block_id}")
                    if self._ref[block_id] == 0:
                        self._free.append(block_id)
                        freed += 1


# ── Task 2: Scheduler ─────────────────────────────────────────────────────────


class Scheduler:
    def __init__(
        self,
        cache_manager: CacheManager,
        block_size: int = BLOCK_SIZE,
        max_seqs: int = MAX_SEQS,
        token_budget: int = TOKEN_BUDGET,
        prefill_chunk: int = PREFILL_CHUNK,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.cache_manager = cache_manager
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.enable_prefix_caching = enable_prefix_caching
        self.scheduling_policy = SchedulingPolicy(scheduling_policy)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.step: int = 0

    def add(self, req: Request) -> None:
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def _preempt(self, req: Request, batch: Batch) -> None:
        """Free req's blocks (respecting lock state), reset its state, re-queue it."""
        if req.cache_handle is not None:
            n = len(req.cache_handle.matched_blocks)
            self.cache_manager.unlock(req.cache_handle)
            self.cache_manager.free(req.block_table[n:])
            req.cache_handle = None
        else:
            self.cache_manager.free(req.block_table)
        req.block_table = []
        req.num_computed_tokens = 0
        req.num_generated_tokens = 0
        req.prefix_tokens_saved = 0
        req.first_token_step = None
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING
        self.running.remove(req)
        self.waiting.appendleft(req)
        batch.preempted.append(req)

    def schedule(self) -> Batch | None:
        """
        Return a single-phase Batch for this step, or None if idle
        (no waiting and no running requests).

        Phase selection policy:
          - PREFILL_FIRST:
              * If any prefill work exists (running prefills or waiting queue
                non-empty), try _schedule_prefill().
              * Otherwise, schedule decode.
          - DECODE_FIRST:
              * If any decode-ready running request exists, try
                _schedule_decode().
              * Otherwise, schedule prefill.

        Delegates to _schedule_prefill() / _schedule_decode().
        See README.md → Task 2 for the full algorithm.
        """
        # TODO
        def has_useful_work(batch: Batch) -> bool:
            return bool(
                batch.to_prefill
                or batch.to_decode
                or batch.preempted
                or batch.newly_admitted
            )

        def has_prefill_work() -> bool:
            return bool(self.waiting) or any(req.is_prefilling for req in self.running)

        def has_decode_work() -> bool:
            return any(not req.is_prefilling for req in self.running)

        try:
            if not self.waiting and not self.running:
                return None

            if self.scheduling_policy == SchedulingPolicy.PREFILL_FIRST:
                if has_prefill_work():
                    batch = self._schedule_prefill()
                    if has_useful_work(batch):
                        return batch
                    if has_decode_work():
                        return self._schedule_decode()
                    return batch
                if has_decode_work():
                    return self._schedule_decode()
                return None

            if has_decode_work():
                batch = self._schedule_decode()
                if has_useful_work(batch):
                    return batch
                if has_prefill_work():
                    return self._schedule_prefill()
                return batch
            if has_prefill_work():
                return self._schedule_prefill()
            return None
        finally:
            self.step += 1

    def _schedule_prefill(self) -> Batch:
        """
        Build a prefill Batch.

        Step A — running requests still prefilling (iterate over a copy - list(self.running)):
          Compute chunk = min(remaining_prefill, prefill_chunk, budget).
          Allocate any new blocks the chunk needs (allocation may evict cache
          entries internally); _preempt on allocation failure.
          Add (req, chunk) to batch.to_prefill; deduct from budget.

        Step B — admit from waiting while budget > 0 and slots remain:
          If prefix caching: call match_prefix FIRST → if hit, lock the
          handle and reduce the number of blocks to allocate.
          Allocate the remaining blocks; on failure unlock the handle and break.
          Build block_table = matched_blocks + newly allocated blocks.
          Set num_computed_tokens, prefix_tokens_saved, cache_handle.
          If the entire prompt was cached, skip adding to to_prefill.
          Append to running and newly_admitted; add first chunk to batch.

        Note:
          Keep this batch phase-pure: populate only batch.to_prefill here.
        """
        # TODO
        batch = Batch(phase=BatchPhase.PREFILL)
        budget = self.token_budget

        for req in list(self.running):
            if budget <= 0:
                break
            if not req.is_prefilling:
                continue

            chunk = min(req.remaining_prefill, self.prefill_chunk, budget)
            if chunk <= 0:
                continue

            needed_blocks = self._blocks_for(req.num_computed_tokens + chunk)
            if needed_blocks > len(req.block_table):
                new_blocks = self.cache_manager.allocate(needed_blocks - len(req.block_table))
                if new_blocks is None:
                    self._preempt(req, batch)
                    continue
                req.block_table.extend(new_blocks)

            batch.to_prefill.append((req, chunk))
            budget -= chunk

        while budget > 0 and self.waiting and len(self.running) < self.max_seqs:
            req = self.waiting[0]
            handle = CacheHandle(0, [])
            if self.enable_prefix_caching:
                handle = self.cache_manager.match_prefix(req.prompt_tokens)
                if handle.num_matched_tokens:
                    self.cache_manager.lock(handle)

            total_blocks = self._blocks_for(req.prompt_len)
            remaining_blocks = total_blocks - len(handle.matched_blocks)
            new_blocks = self.cache_manager.allocate(remaining_blocks)
            if new_blocks is None:
                if handle.num_matched_tokens:
                    self.cache_manager.unlock(handle)
                break

            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            req.block_table = list(handle.matched_blocks) + new_blocks
            req.num_computed_tokens = handle.num_matched_tokens
            req.num_generated_tokens = 0
            req.prefix_tokens_saved = handle.num_matched_tokens
            req.cache_handle = handle if handle.num_matched_tokens else None
            if req.first_scheduled_step is None:
                req.first_scheduled_step = self.step

            self.running.append(req)
            batch.newly_admitted.append(req)

            if req.is_prefilling and budget > 0:
                chunk = min(req.remaining_prefill, self.prefill_chunk, budget)
                if chunk > 0:
                    batch.to_prefill.append((req, chunk))
                    budget -= chunk

        return batch

    def _schedule_decode(self) -> Batch:
        """
        Build a decode Batch (iterate over a copy of running).

        For each request: if the next token crosses a block boundary
        (tokens_so_far + 1 needs a new block), allocate one block;
        _preempt on failure. Append to batch.to_decode.

        Note:
          Only include decode-ready requests (not still-prefilling ones).
        """
        # TODO
        batch = Batch(phase=BatchPhase.DECODE)

        for req in list(self.running):
            if req.is_prefilling:
                continue

            tokens_after_decode = req.num_computed_tokens + req.num_generated_tokens + 1
            needed_blocks = self._blocks_for(tokens_after_decode)
            if needed_blocks > len(req.block_table):
                new_block = self.cache_manager.allocate(1)
                if new_block is None:
                    self._preempt(req, batch)
                    continue
                req.block_table.extend(new_block)

            batch.to_decode.append(req)

        return batch


# ── MiniEngine (provided — do not modify) ────────────────────────────────────


class MiniEngine:
    def __init__(
        self,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.enable_prefix_caching = enable_prefix_caching
        self.cache_manager = CacheManager(num_blocks, block_size)
        self.model = DummyLLM(num_blocks, block_size)
        self.scheduler = Scheduler(
            self.cache_manager,
            block_size,
            enable_prefix_caching=enable_prefix_caching,
            scheduling_policy=scheduling_policy,
        )

    def run(
        self, workload: list[Request], label: str = ""
    ) -> tuple[list[Request], list[StepMetrics]]:
        requests = sorted([r.copy() for r in workload], key=lambda r: r.arrival_step)
        finished: list[Request] = []
        all_metrics: list[StepMetrics] = []
        next_idx, step = 0, 0
        prog = tqdm(desc=label, unit="step", mininterval=0.25)
        last_prog_ts = 0.0

        def refresh_progress(force: bool = False) -> None:
            nonlocal last_prog_ts
            now = time.monotonic()
            if force or (now - last_prog_ts >= 0.5):
                prog.update(step - prog.n)
                prog.set_postfix_str(
                    f"done={len(finished)}/{len(requests)} "
                    f"running={len(self.scheduler.running)} "
                    f"waiting={len(self.scheduler.waiting)}"
                )
                last_prog_ts = now

        while len(finished) < len(requests):
            # Admit newly arrived requests
            while next_idx < len(requests) and requests[next_idx].arrival_step <= step:
                self.scheduler.add(requests[next_idx])
                next_idx += 1

            if not self.scheduler.running and not self.scheduler.waiting:
                if next_idx < len(requests):
                    step = requests[next_idx].arrival_step
                    continue
                break

            batch = self.scheduler.schedule()
            if batch is None:
                step += 1
                refresh_progress()
                continue

            if batch.is_prefill:
                for req, chunk in batch.to_prefill:
                    req._next_token = self.model.prefill(
                        req.prompt_tokens,
                        req.block_table,
                        req.num_computed_tokens,
                        chunk,
                    )
                    req.num_computed_tokens += chunk
            else:
                for req in batch.to_decode:
                    input_tok = getattr(req, "_next_token", req.prompt_tokens[-1])
                    pos = req.num_computed_tokens + req.num_generated_tokens
                    req._next_token = self.model.decode(
                        input_tok,
                        req.block_table,
                        pos,
                    )
                    req.num_generated_tokens += 1
                    if req.num_generated_tokens == 1 and req.first_token_step is None:
                        req.first_token_step = step

            done_this_step = 0
            for req in list(self.scheduler.running):
                if req.is_done:
                    req.finish_step = step
                    req.status = RequestStatus.DONE
                    self.scheduler.running.remove(req)
                    if self.enable_prefix_caching:
                        self.cache_manager.insert_prefix(
                            req.prompt_tokens, req.block_table
                        )
                    if req.cache_handle is not None:
                        n = len(req.cache_handle.matched_blocks)
                        self.cache_manager.unlock(req.cache_handle)
                        self.cache_manager.free(req.block_table[n:])
                    else:
                        self.cache_manager.free(req.block_table)
                    finished.append(req)
                    done_this_step += 1
            all_metrics.append(
                StepMetrics(
                    step=step,
                    decode_tokens=len(batch.to_decode),
                    prefill_tokens=sum(c for _, c in batch.to_prefill),
                    num_running=len(self.scheduler.running),
                    num_waiting=len(self.scheduler.waiting),
                    kv_blocks_used=self.cache_manager.num_blocks
                    - self.cache_manager.num_free_blocks,
                    prefix_tokens_saved=sum(
                        r.prefix_tokens_saved for r in batch.newly_admitted
                    ),
                )
            )
            step += 1
            refresh_progress(force=done_this_step > 0)

        refresh_progress(force=True)
        prog.close()
        return finished, all_metrics


# ── Main (provided — do not modify) ──────────────────────────────────────────


def main():
    print("=" * 60)
    print("HW3: Mini Inference Engine")
    print("=" * 60)

    workload_configs = [
        (
            "Prefill-Heavy",
            dict(
                prompt_len_range=(64, 256),
                output_len_range=(30, 150),
                shared_prefix_len=256,
            ),
        ),
        (
            "Decode-Heavy",
            dict(
                num_requests=50,
                prompt_len_range=(48, 128),
                output_len_range=(150, 400),
                shared_prefix_len=32,
            ),
        ),
    ]

    all_results: list[tuple] = []
    policy_results: list[tuple] = []
    for label, wl_kwargs in workload_configs:
        wl = generate_workload(**wl_kwargs)
        print(f"\n{'─' * 60}")
        print(f"  {label}  ({len(wl)} requests)\n")

        eng_off = MiniEngine(enable_prefix_caching=False)
        fin_off, met_off = eng_off.run(wl, label="no-cache")
        stats_off = compute_stats(fin_off, met_off, len(met_off))
        print_stats("No prefix cache", stats_off)

        eng_on = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
        )
        fin_on, met_on = eng_on.run(wl, label="cache-on")
        stats_on = compute_stats(fin_on, met_on, len(met_on))
        print_stats("Prefix cache ON", stats_on)

        speedup = stats_off["total_steps"] / max(stats_on["total_steps"], 1)
        print(
            f"\n    Steps: {stats_off['total_steps']} → {stats_on['total_steps']}  "
            f"({speedup:.2f}× fewer)"
        )
        print(f"    TTFT:  {stats_off['ttft_mean']} → {stats_on['ttft_mean']} steps")

        all_results.append((label, met_off, met_on, stats_off, stats_on))

        eng_decode_first = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.DECODE_FIRST,
        )
        fin_df, met_df = eng_decode_first.run(wl, label="cache-on/decode-first")
        stats_df = compute_stats(fin_df, met_df, len(met_df))

        print("\n  Scheduling policy (cache ON)")
        print(
            f"    Prefill-first steps / TTFT / E2E : "
            f"{stats_on['total_steps']} / {stats_on['ttft_mean']} / {stats_on['e2e_mean']}"
        )
        print(
            f"    Decode-first  steps / TTFT / E2E : "
            f"{stats_df['total_steps']} / {stats_df['ttft_mean']} / {stats_df['e2e_mean']}"
        )
        policy_results.append((label, met_on, met_df, stats_on, stats_df))

    print(f"\n{'─' * 60}")
    plot_results(all_results)
    plot_policy_results(policy_results)


if __name__ == "__main__":
    main()


# ── Writeup ───────────────────────────────────────────────────────────────────
#
# Q1: Compare the prefix cache's impact on TTFT and E2E latency between the
#     two workloads.  Why is the speedup much larger for the prefill-heavy
#     workload?  Give specific numbers from your run.
#
# Q2: Trace the ref-count lifecycle of a shared prefix block from the moment
#     a first request finishes (insert_prefix) through a second request
#     using that block (match_prefix → lock → run → unlock) to the eventual
#     eviction.  What is the ref count at each stage, and what prevents the
#     block from being evicted while the second request is live?
#
# Q3: With prefix caching ON, why does eviction reduce preemptions compared
#     to the no-caching run?  Under what condition would eviction fail and
#     fall back to preemption?
#
# Q4: Compare the two scheduling policies (PREFILL_FIRST vs DECODE_FIRST)
#     using the numbers on your policy-comparison plot. On which workload
#     does the choice of policy matter a lot, and on which is it almost
#     a wash?  Explain what each policy optimises for, and name a
#     realistic scenario in which you would pick each one.
#
# Q1:
# Prefix caching helped both workloads in the latest collected run, but much more on
# Prefill-Heavy. Prefill-Heavy dropped from 697 to 284 total steps, TTFT mean
# dropped from 233.0 to 48.1 steps, and E2E mean dropped from 360.6 to 152.3
# steps. The cache saved 11008 prompt tokens with a 50.1% hit rate.
# Decode-Heavy improved less: 1511 to 1095 total steps, TTFT mean 737.7 to
# 430.8, and E2E mean 1084.2 to 773.1, with 1920 tokens saved and a 10.5% hit
# rate. The prefill-heavy workload has long shared prompts, so prefix reuse
# removes a large fraction of the expensive prompt computation; decode-heavy
# spends more of its time generating new tokens that cannot be skipped by
# prefix caching.
#
# Q2:
# When the first request finishes, insert_prefix() creates cache entries for
# complete prompt blocks while the request still owns them. For a shared block,
# ref goes from 1 request owner to 2 when the cache first adopts it; overlapping
# cache entries increase cache_ref but do not keep increasing ref. When the
# finished request frees its ownership, ref drops to 1, meaning cached-only and
# evictable. A later request calls match_prefix(), which only returns a handle;
# lock(handle) then pins the matched blocks and raises ref to 2 while the
# request is live. During that time eviction skips the block because it only
# evicts entries whose blocks all have ref == 1. When the request finishes,
# unlock(handle) drops ref back to 1. A later LRU eviction removes cache entries,
# decrements cache_ref, and when cache_ref reaches 0 the cache ownership is
# released so ref drops to 0 and the block returns to the free list.
#
# Q3:
# With caching ON, eviction can reclaim cached-only prefix blocks before the
# scheduler preempts live requests. That reduced preemptions in the collected
# run:
# Prefill-Heavy went from 91 preemptions without caching to 18 with caching,
# and Decode-Heavy went from 273 to 159. Eviction fails when there are not
# enough cached-only blocks to reclaim, either because blocks are still pinned
# by live requests (ref > 1) or because the cache does not contain enough
# evictable entries. At that point allocation returns None and the scheduler
# falls back to preempting a running request.
#
# Q4:
# With prefix caching ON, policy mattered a lot on Prefill-Heavy: prefill-first
# finished in 284 steps with TTFT/E2E means of 48.1/152.3, while decode-first
# took 409 steps with 100.4/191.6. On Decode-Heavy it was almost a wash for
# total steps: prefill-first took 1095 steps and decode-first took 1105; TTFT
# was also close at 430.8 vs 434.2, though decode-first had lower E2E mean
# (731.5 vs 773.1). Prefill-first optimizes admission and time-to-first-token
# by quickly computing prompts and exposing more requests to batching. Decode-
# first optimizes progress for requests that are already generating, which can
# reduce tail/E2E latency when interactive users are waiting on ongoing streams.
# The latest HW3 validation run passed all 45 tests.
#
