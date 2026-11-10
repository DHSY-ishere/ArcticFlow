package router

import (
	"container/list"
	"math/bits"
	"sync"
)

// DefaultBlockSize is the KV-cache page size in tokens (vLLM's default).
const DefaultBlockSize = 16

// MaxWorkers is bounded by the uint64 residency bitmask on every node.
const MaxWorkers = 64

const (
	fnvOffset64 = 14695981039346656037
	fnvPrime64  = 1099511628211
)

// HashBlocks splits tokens into full blocks of blockSize and returns the
// *chained* FNV-1a 64 hash of every block: h_i = FNV1a(h_{i-1} || block_i).
// Chaining makes h_i a commitment to the entire prefix [0, (i+1)*blockSize),
// so equal hashes imply equal prefixes (up to 64-bit collisions). A trailing
// partial block is ignored: partial pages are never shareable in a paged KV
// cache.
func HashBlocks(tokens []uint32, blockSize int) []uint64 {
	n := len(tokens) / blockSize
	out := make([]uint64, n)
	h := uint64(fnvOffset64)
	for i := 0; i < n; i++ {
		for _, t := range tokens[i*blockSize : (i+1)*blockSize] {
			h ^= uint64(byte(t))
			h *= fnvPrime64
			h ^= uint64(byte(t >> 8))
			h *= fnvPrime64
			h ^= uint64(byte(t >> 16))
			h *= fnvPrime64
			h ^= uint64(byte(t >> 24))
			h *= fnvPrime64
		}
		out[i] = h
	}
	return out
}

type node struct {
	hash     uint64
	parent   *node
	children map[uint64]*node
	holders  uint64                // bit w set <=> worker w holds this block
	elems    map[int]*list.Element // worker -> position in that worker's LRU
}

// RadixCache is a shadow index of every worker's paged KV cache: a radix tree
// (trie over the alphabet of chained block hashes) whose nodes record which
// workers currently hold the corresponding KV page. Reads take an RLock and
// never mutate; inserts/evictions take the write lock.
//
// Eviction is per-worker LRU over blocks with a leaf-first invariant: a path
// is always touched deepest-block-first so that parents are strictly more
// recently used than their children, hence the LRU tail is always a leaf for
// that worker (matching vLLM/SGLang leaf eviction).
type RadixCache struct {
	mu        sync.RWMutex
	blockSize int
	root      *node
	lrus      []*list.List
	capacity  []int // max blocks per worker
	nodes     int
	evictions uint64
}

// NewRadixCache creates a cache for numWorkers workers, each holding at most
// capacityBlocks blocks (blockSize tokens each).
func NewRadixCache(numWorkers, blockSize, capacityBlocks int) *RadixCache {
	if numWorkers > MaxWorkers {
		panic("router: too many workers for residency bitmask")
	}
	c := &RadixCache{
		blockSize: blockSize,
		root:      &node{children: map[uint64]*node{}},
		lrus:      make([]*list.List, numWorkers),
		capacity:  make([]int, numWorkers),
	}
	for i := range c.lrus {
		c.lrus[i] = list.New()
		c.capacity[i] = capacityBlocks
	}
	return c
}

// BlockSize returns the page size in tokens.
func (c *RadixCache) BlockSize() int { return c.blockSize }

// Match returns, for each worker, the number of leading blocks of `hashes`
// that the worker holds contiguously from the root. Only workers whose bit is
// set in `mask` are considered. Complexity O(depth) regardless of worker count.
func (c *RadixCache) Match(hashes []uint64, mask uint64) []int {
	out := make([]int, len(c.lrus))
	c.mu.RLock()
	defer c.mu.RUnlock()
	cur := c.root
	alive := mask
	depth := 0
	for _, h := range hashes {
		child, ok := cur.children[h]
		if !ok {
			break
		}
		depth++
		next := alive & child.holders
		for dropped := alive &^ next; dropped != 0; dropped &= dropped - 1 {
			out[bits.TrailingZeros64(dropped)] = depth - 1
		}
		alive = next
		if alive == 0 {
			return out
		}
		cur = child
	}
	for ; alive != 0; alive &= alive - 1 {
		out[bits.TrailingZeros64(alive)] = depth
	}
	return out
}

// Insert records that `worker` now holds every block on the path `hashes`
// (creating nodes as needed), refreshes LRU recency, then evicts to capacity.
// It returns how many blocks were newly resident and how many were evicted.
func (c *RadixCache) Insert(worker int, hashes []uint64) (added, evicted int) {
	if len(hashes) == 0 {
		return 0, 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	path := make([]*node, 0, len(hashes))
	cur := c.root
	for _, h := range hashes {
		child, ok := cur.children[h]
		if !ok {
			child = &node{hash: h, parent: cur, children: map[uint64]*node{}}
			cur.children[h] = child
			c.nodes++
		}
		path = append(path, child)
		cur = child
	}
	bit := uint64(1) << uint(worker)
	lru := c.lrus[worker]
	// Deepest first => root-most block ends up most-recently-used.
	for i := len(path) - 1; i >= 0; i-- {
		n := path[i]
		if n.holders&bit != 0 {
			lru.MoveToFront(n.elems[worker])
			continue
		}
		if n.elems == nil {
			n.elems = make(map[int]*list.Element, 2)
		}
		n.holders |= bit
		n.elems[worker] = lru.PushFront(n)
		added++
	}
	evicted = c.evictLocked(worker)
	return added, evicted
}

func (c *RadixCache) evictLocked(worker int) int {
	lru := c.lrus[worker]
	bit := uint64(1) << uint(worker)
	n := 0
	for lru.Len() > c.capacity[worker] {
		el := lru.Back()
		nd := el.Value.(*node)
		lru.Remove(el)
		delete(nd.elems, worker)
		nd.holders &^= bit
		c.pruneLocked(nd)
		n++
	}
	c.evictions += uint64(n)
	return n
}

// pruneLocked unlinks nodes that no worker holds and that have no children.
func (c *RadixCache) pruneLocked(n *node) {
	for n != c.root && n.holders == 0 && len(n.children) == 0 {
		delete(n.parent.children, n.hash)
		c.nodes--
		n = n.parent
	}
}

// RemoveWorker drops all residency for a worker (crash / restart / drain).
func (c *RadixCache) RemoveWorker(worker int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	lru := c.lrus[worker]
	bit := uint64(1) << uint(worker)
	for el := lru.Back(); el != nil; el = lru.Back() {
		nd := el.Value.(*node)
		lru.Remove(el)
		delete(nd.elems, worker)
		nd.holders &^= bit
		c.pruneLocked(nd)
	}
}

// Stats is a point-in-time snapshot of the index.
type CacheStats struct {
	Nodes          int
	Evictions      uint64
	BlocksByWorker []int
}

// Stats returns a snapshot for observability.
func (c *RadixCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s := CacheStats{Nodes: c.nodes, Evictions: c.evictions, BlocksByWorker: make([]int, len(c.lrus))}
	for i, l := range c.lrus {
		s.BlocksByWorker[i] = l.Len()
	}
	return s
}
