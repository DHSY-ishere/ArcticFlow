package router

import (
	"hash/fnv"
	"regexp"
	"strings"
)

// VocabSize is the size of the hashed pseudo-vocabulary. It matches the
// Qwen2.5 vocabulary size so simulated and real token-id ranges are alike.
const VocabSize = 151643

// pieceRe splits text into GPT-style pieces: an optional single leading
// whitespace glued to a word or a single punctuation rune, or a whitespace run.
// The identical expression lives in engine/tokenizer.py; the two tokenizers
// MUST stay byte-for-byte compatible so router and worker agree on block hashes.
var pieceRe = regexp.MustCompile(`[ \t\n\f\r]?[A-Za-z0-9_]+|[ \t\n\f\r]?[^ \t\n\f\r A-Za-z0-9_]|[ \t\n\f\r]+`)

// Tokenize converts text into deterministic pseudo token ids (FNV-1a 32 of the
// piece, modulo VocabSize). It is a routing tokenizer: it only needs to be
// stable and roughly token-count-faithful, not model-exact.
func Tokenize(text string) []uint32 {
	pieces := pieceRe.FindAllString(text, -1)
	out := make([]uint32, len(pieces))
	for i, p := range pieces {
		h := fnv.New32a()
		_, _ = h.Write([]byte(p))
		out[i] = h.Sum32() % VocabSize
	}
	return out
}

// Message is one conversational turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

const (
	promptPreamble = "You are ArcticFlow Analyst, a text-to-SQL engine. Generate exactly one read-only DuckDB SELECT statement grounded strictly in the semantic model below.\n\n### SEMANTIC MODEL\n"
	promptConvo    = "\n### CONVERSATION\n"
)

// BuildPrompt renders the canonical prompt. The semantic model (large, shared,
// stable) is placed FIRST so that every request against the same model shares
// a long common prefix — the property RadixAttention-style routing exploits.
func BuildPrompt(semanticModel string, msgs []Message) string {
	var b strings.Builder
	b.Grow(len(promptPreamble) + len(semanticModel) + 256)
	b.WriteString(promptPreamble)
	b.WriteString(semanticModel)
	if !strings.HasSuffix(semanticModel, "\n") {
		b.WriteByte('\n')
	}
	b.WriteString(promptConvo)
	for _, m := range msgs {
		switch m.Role {
		case "assistant":
			b.WriteString("ASSISTANT: ")
		default:
			b.WriteString("USER: ")
		}
		b.WriteString(m.Content)
		b.WriteByte('\n')
	}
	b.WriteString("ASSISTANT:")
	return b.String()
}
