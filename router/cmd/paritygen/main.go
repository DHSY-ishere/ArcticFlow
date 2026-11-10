// Command paritygen writes golden tokenizer/block-hash vectors consumed by
// engine/tests/test_engine.py::test_go_python_parity, proving router and worker agree.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/arcticflow/arcticflow/router"
)

func main() {
	text := router.BuildPrompt(strings.Repeat("tables:\n  - name: orders # café ünïcode → ok\n    dimensions:\n      - {name: a, expr: \"x.y\"}\n", 40),
		[]router.Message{{Role: "user", Content: "What's total revenue by region, top 5?"}})
	toks := router.Tokenize(text)
	hs := router.HashBlocks(toks, router.DefaultBlockSize)
	hstr := make([]string, len(hs))
	for i, h := range hs {
		hstr[i] = fmt.Sprint(h)
	}
	out := os.Args[1]
	b, _ := json.Marshal(map[string]any{"text": text, "tokens": toks, "hashes": hstr})
	if err := os.WriteFile(out, b, 0o644); err != nil {
		panic(err)
	}
}
