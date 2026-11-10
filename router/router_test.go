package router

import (
	"sync"
	"testing"
)

func seq(start, n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(start + i)
	}
	return out
}

func TestHashBlocksChained(t *testing.T) {
	a := HashBlocks(seq(0, 48), 16)
	b := HashBlocks(append(seq(0, 32), seq(1000, 16)...), 16)
	if len(a) != 3 || len(b) != 3 {
		t.Fatalf("want 3 blocks, got %d %d", len(a), len(b))
	}
	if a[0] != b[0] || a[1] != b[1] || a[2] == b[2] {
		t.Fatalf("shared 2-block prefix must hash equal, diverging block must differ")
	}
	// Same block content at different positions must NOT collide (chaining).
	c := HashBlocks(append(seq(500, 16), seq(0, 16)...), 16)
	if c[1] == a[0] {
		t.Fatal("chain hash must commit to the prefix")
	}
	if got := len(HashBlocks(seq(0, 47), 16)); got != 2 {
		t.Fatalf("partial block must be ignored, got %d", got)
	}
}

func TestMatchAndContiguity(t *testing.T) {
	c := NewRadixCache(3, 16, 1000)
	p := HashBlocks(seq(0, 160), 16) // 10 blocks
	c.Insert(0, p)
	c.Insert(1, p[:4])
	m := c.Match(p, 0b111)
	if m[0] != 10 || m[1] != 4 || m[2] != 0 {
		t.Fatalf("unexpected overlap %v", m)
	}
	// Mask excludes worker 0.
	m = c.Match(p, 0b110)
	if m[0] != 0 || m[1] != 4 {
		t.Fatalf("mask ignored: %v", m)
	}
	// Diverging suffix.
	q := HashBlocks(append(seq(0, 64), seq(9000, 64)...), 16)
	m = c.Match(q, 0b111)
	if m[0] != 4 || m[1] != 4 {
		t.Fatalf("divergence overlap wrong: %v", m)
	}
}

func TestLRUEvictsLeavesFirst(t *testing.T) {
	c := NewRadixCache(1, 16, 8)
	a := HashBlocks(seq(0, 16*6), 16)     // path A: 6 blocks
	b := HashBlocks(seq(10000, 16*6), 16) // path B: 6 blocks, disjoint
	c.Insert(0, a)
	_, ev := c.Insert(0, b) // 12 > 8 => evict 4 from A's tail (oldest leaves)
	if ev != 4 {
		t.Fatalf("want 4 evictions, got %d", ev)
	}
	if got := c.Match(a, 1)[0]; got != 2 {
		t.Fatalf("A should keep its 2-block root prefix, got %d", got)
	}
	if got := c.Match(b, 1)[0]; got != 6 {
		t.Fatalf("B should be fully resident, got %d", got)
	}
	// Re-touching A's prefix protects it from the next eviction.
	c.Insert(0, a[:2])
	d := HashBlocks(seq(20000, 16*4), 16)
	c.Insert(0, d)
	if got := c.Match(a, 1)[0]; got != 2 {
		t.Fatalf("touched prefix evicted: %d", got)
	}
	if st := c.Stats(); st.BlocksByWorker[0] != 8 {
		t.Fatalf("capacity violated: %v", st.BlocksByWorker)
	}
}

func TestPruneAndRemoveWorker(t *testing.T) {
	c := NewRadixCache(2, 16, 4)
	p := HashBlocks(seq(0, 16*4), 16)
	c.Insert(0, p)
	c.Insert(1, p)
	c.RemoveWorker(0)
	if m := c.Match(p, 0b11); m[0] != 0 || m[1] != 4 {
		t.Fatalf("after remove: %v", m)
	}
	c.RemoveWorker(1)
	if n := c.Stats().Nodes; n != 0 {
		t.Fatalf("tree should be fully pruned, %d nodes left", n)
	}
}

func TestConcurrentAccess(t *testing.T) {
	c := NewRadixCache(4, 16, 64)
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 300; i++ {
				h := HashBlocks(seq((g*7+i)%40*1000, 16*8), 16)
				c.Insert(g%4, h)
				c.Match(h, 0xF)
			}
		}(g)
	}
	wg.Wait()
	for _, n := range c.Stats().BlocksByWorker {
		if n > 64 {
			t.Fatalf("capacity violated under concurrency: %d", n)
		}
	}
}

func mkWorkers(n int) []*Worker {
	ws := make([]*Worker, n)
	for i := range ws {
		ws[i] = NewWorker(i, "w", "http://x")
	}
	return ws
}

func TestBalancerPrefersOverlapThenSheds(t *testing.T) {
	ws := mkWorkers(3)
	c := NewRadixCache(3, 16, 10000)
	b := NewKVAwareBalancer(c, ws, 1.0, 0.05)
	prompt := seq(0, 16*20)

	d0, _ := b.Route(prompt, 100, 0) // cold: goes somewhere, becomes the owner
	owner := d0.Worker.ID
	// Followers with the same prefix stick to the owner while its queue is shallow.
	for i := 0; i < 10; i++ {
		d, _ := b.Route(append(append([]uint32{}, prompt...), uint32(70000+i)), 100, 0)
		if d.Worker.ID != owner || d.Ratio < 0.9 {
			t.Fatalf("expected affinity to worker %d, got %d ratio %.2f", owner, d.Worker.ID, d.Ratio)
		}
	}
	// Owner queue is now 11 deep => penalty 0.55 < alpha, still sticky. Push past 1/beta=20.
	for i := 0; i < 15; i++ {
		_, _ = b.Route(append(append([]uint32{}, prompt...), uint32(80000+i)), 100, 0)
	}
	spill := 0
	for i := 0; i < 10; i++ {
		d, _ := b.Route(append(append([]uint32{}, prompt...), uint32(90000+i)), 100, 0)
		if d.Worker.ID != owner {
			spill++
		}
	}
	if spill == 0 {
		t.Fatal("hot-spot penalty never shed load to another worker")
	}
}

func TestBalancerExcludeAndHealth(t *testing.T) {
	ws := mkWorkers(2)
	c := NewRadixCache(2, 16, 100)
	b := NewKVAwareBalancer(c, ws, 1, 0.01)
	ws[0].SetHealthy(false)
	d, err := b.Route(seq(0, 32), 1, 0)
	if err != nil || d.Worker.ID != 1 {
		t.Fatalf("unhealthy worker routed: %v %v", d.Worker, err)
	}
	if _, err = b.Route(seq(0, 32), 1, 1<<1); err != ErrNoWorkers {
		t.Fatalf("want ErrNoWorkers, got %v", err)
	}
	b.Release(d)
	if ws[1].QueueDepth() != 0 || ws[1].InFlightTokens() != 0 {
		t.Fatal("release did not restore counters")
	}
}

func TestRoundRobinFair(t *testing.T) {
	ws := mkWorkers(4)
	rr := NewRoundRobin(ws)
	cnt := map[int]int{}
	for i := 0; i < 400; i++ {
		d, _ := rr.Route(nil, 0, 0)
		cnt[d.Worker.ID]++
	}
	for id, n := range cnt {
		if n != 100 {
			t.Fatalf("worker %d got %d", id, n)
		}
	}
}

func TestTokenizerStable(t *testing.T) {
	a := Tokenize("SELECT customers.region, SUM(orders.amount) AS rev FROM orders")
	b := Tokenize("SELECT customers.region, SUM(orders.amount) AS rev FROM orders")
	if len(a) == 0 || len(a) != len(b) {
		t.Fatal("tokenizer unstable")
	}
	for i := range a {
		if a[i] != b[i] || a[i] >= VocabSize {
			t.Fatal("tokenizer unstable/out of range")
		}
	}
}

func BenchmarkMatch(b *testing.B) {
	c := NewRadixCache(8, 16, 1<<20)
	for m := 0; m < 20; m++ {
		c.Insert(m%8, HashBlocks(seq(m*100000, 16*500), 16)) // 8k-token models
	}
	q := HashBlocks(seq(5*100000, 16*500), 16)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			c.Match(q, 0xFF)
		}
	})
}

func BenchmarkRoute(b *testing.B) {
	ws := mkWorkers(8)
	c := NewRadixCache(8, 16, 1<<20)
	bal := NewKVAwareBalancer(c, ws, 1, 0.01)
	toks := seq(0, 16*500)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			d, _ := bal.Route(toks, 64, 0)
			bal.Release(d)
		}
	})
}
