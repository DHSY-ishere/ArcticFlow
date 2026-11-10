"""Paged KV-cache block manager with chained-hash prefix sharing, ref-counting and LRU eviction.

Memory model (mirrors vLLM's automatic prefix caching):
  * GPU KV memory is `total_blocks` pages of `block_size` tokens.
  * A *cached* block is a full prompt page identified by its chained prefix hash. It may be shared
    by many running requests (ref > 0, pinned) or idle (ref == 0, evictable).
  * A running request additionally holds *private* blocks for its uncached prompt tail + decode
    growth. Private blocks are reserved at admission (no preemption), so admission can fail and the
    request waits - exactly the back-pressure a real engine exhibits.
  * Eviction is LRU over idle blocks, leaf-first (a block is idle-released deepest-first, so the
    parent is always younger than its children and the LRU head is always a leaf).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class _Block:
    h: int
    parent: int | None
    ref: int = 0


@dataclass
class Allocation:
    cached_blocks: list[int]            # hashes of matched (shared) prefix blocks, pinned
    private: int                        # private blocks reserved
    committed: list[int] = field(default_factory=list)   # hashes this request added to the cache
    prompt_hashes: list[int] = field(default_factory=list)


class KVBlockManager:
    def __init__(self, total_blocks: int, block_size: int = 16, kv_bytes_per_token: int = 28672):
        self.total, self.bs, self.kv_bpt = total_blocks, block_size, kv_bytes_per_token
        self.blocks: dict[int, _Block] = {}
        self.idle: OrderedDict[int, None] = OrderedDict()    # evictable (ref == 0), oldest first
        self.private_total = 0
        self.hit_tokens = 0
        self.lookup_tokens = 0
        self.evictions = 0

    # -- accounting ------------------------------------------------------------------------------
    @property
    def used(self) -> int:
        return len(self.blocks) + self.private_total

    @property
    def free(self) -> int:
        return self.total - self.used

    @property
    def reclaimable(self) -> int:
        return self.free + len(self.idle)

    # -- prefix lookup ---------------------------------------------------------------------------
    def peek(self, hashes: list[int]) -> int:
        """Number of leading blocks currently cached (no side effects)."""
        n = 0
        for h in hashes:
            if h not in self.blocks:
                break
            n += 1
        return n

    def admit(self, hashes: list[int], prompt_len: int, max_new: int) -> Allocation | None:
        """Pin the cached prefix and reserve private blocks for the rest. None => no room (wait)."""
        n = self.peek(hashes)
        need = max(0, -(-(prompt_len + max_new) // self.bs) - n)
        # Pinning matched blocks removes them from the evictable set, so account for that first.
        newly_pinned = sum(1 for h in hashes[:n] if self.blocks[h].ref == 0)
        if need > self.free + len(self.idle) - newly_pinned:
            return None
        for h in hashes[:n]:
            b = self.blocks[h]
            if b.ref == 0:
                self.idle.pop(h, None)
            b.ref += 1
        self._make_room(need)
        self.private_total += need
        self.lookup_tokens += prompt_len
        self.hit_tokens += n * self.bs
        return Allocation(list(hashes[:n]), need, prompt_hashes=hashes)

    def _make_room(self, need: int) -> None:
        while self.free < need:
            h, _ = self.idle.popitem(last=False)
            b = self.blocks.pop(h)
            assert b.ref == 0
            self.evictions += 1

    def commit_prefill(self, alloc: Allocation) -> None:
        """Prompt fully computed: turn the request's private prompt pages into shared cached pages."""
        n_cached = len(alloc.cached_blocks)
        prev = alloc.cached_blocks[-1] if alloc.cached_blocks else None
        for h in alloc.prompt_hashes[n_cached:]:
            if h in self.blocks:            # raced with a concurrent identical prefill: share it
                self.blocks[h].ref += 1
                if self.blocks[h].ref == 1:
                    self.idle.pop(h, None)
                self.private_total -= 1
            else:
                self.blocks[h] = _Block(h, prev, ref=1)
                self.private_total -= 1
            alloc.committed.append(h)
            prev = h

    def release(self, alloc: Allocation) -> None:
        """Request finished: free private pages, unpin shared ones (deepest first => leaf-first LRU)."""
        remaining_private = alloc.private - len(alloc.committed)
        self.private_total -= remaining_private
        for h in reversed(alloc.cached_blocks + alloc.committed):
            b = self.blocks.get(h)
            if b is None:
                continue
            b.ref -= 1
            if b.ref == 0:
                self.idle[h] = None
                self.idle.move_to_end(h)

    def stats(self) -> dict:
        return {"total_blocks": self.total, "cached_blocks": len(self.blocks), "idle_blocks": len(self.idle),
                "private_blocks": self.private_total, "free_blocks": self.free,
                "utilization": round(self.used / self.total, 4), "evictions": self.evictions,
                "hit_ratio": round(self.hit_tokens / self.lookup_tokens, 4) if self.lookup_tokens else 0.0,
                "kv_bytes_used": self.used * self.bs * self.kv_bpt}
