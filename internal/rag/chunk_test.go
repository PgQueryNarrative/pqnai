package rag

import (
	"strings"
	"testing"
)

func TestChunkValidation(t *testing.T) {
	cases := []struct {
		name string
		opts ChunkOptions
	}{
		{"zero size", ChunkOptions{Size: 0, Overlap: 0}},
		{"negative size", ChunkOptions{Size: -1, Overlap: 0}},
		{"negative overlap", ChunkOptions{Size: 10, Overlap: -1}},
		{"overlap equal to size", ChunkOptions{Size: 10, Overlap: 10}},
		{"overlap greater than size", ChunkOptions{Size: 10, Overlap: 20}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Chunk("some text", c.opts); err == nil {
				t.Fatalf("expected error for %s", c.name)
			}
		})
	}
}

func TestChunkEmptyInput(t *testing.T) {
	for _, text := range []string{"", "   ", "\n\n\t"} {
		chunks, err := Chunk(text, ChunkOptions{Size: 10, Overlap: 2})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(chunks) != 0 {
			t.Fatalf("expected no chunks for %q, got %v", text, chunks)
		}
	}
}

func TestChunkShorterThanSize(t *testing.T) {
	text := "a short piece of text"
	chunks, err := Chunk(text, ChunkOptions{Size: 100, Overlap: 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 || chunks[0] != text {
		t.Fatalf("expected single unchanged chunk, got %v", chunks)
	}
}

func TestChunkExactlyAtSize(t *testing.T) {
	text := strings.Repeat("a", 50)
	chunks, err := Chunk(text, ChunkOptions{Size: 50, Overlap: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected single chunk when text length == size, got %d", len(chunks))
	}
}

func TestChunkMultipleWithParagraphBreaks(t *testing.T) {
	para1 := strings.Repeat("alpha ", 10)   // 60 runes
	para2 := strings.Repeat("bravo ", 10)   // 60 runes
	para3 := strings.Repeat("charlie ", 10) // 80 runes
	text := para1 + "\n\n" + para2 + "\n\n" + para3

	chunks, err := Chunk(text, ChunkOptions{Size: 70, Overlap: 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d: %v", len(chunks), chunks)
	}
	for _, c := range chunks {
		if strings.Contains(c, "\n\n") {
			t.Fatalf("chunk should not contain a paragraph break, got %q", c)
		}
	}
}

func TestChunkOverlapActuallyOverlaps(t *testing.T) {
	text := strings.Repeat("word ", 100) // 500 runes, no natural break points except spaces
	chunks, err := Chunk(text, ChunkOptions{Size: 100, Overlap: 20})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}

	// The tail of each chunk should share content with the head of the next.
	for i := 0; i < len(chunks)-1; i++ {
		tail := lastNRunes(chunks[i], 10)
		if !strings.Contains(chunks[i+1], tail) {
			t.Fatalf("chunk %d tail %q not found in chunk %d %q", i, tail, i+1, chunks[i+1])
		}
	}
}

func TestChunkForwardProgressGuaranteed(t *testing.T) {
	// A pathological input with no whitespace at all, forcing hard cuts.
	text := strings.Repeat("x", 1000)
	chunks, err := Chunk(text, ChunkOptions{Size: 50, Overlap: 49})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	// Must terminate (the test itself times out if it doesn't) and cover
	// the whole input.
	joined := strings.Join(chunks, "")
	if len(joined) < len(text) {
		t.Fatalf("chunks do not appear to cover the full input: got %d runes total", len(joined))
	}
}

func TestChunkUnicodeSafety(t *testing.T) {
	text := strings.Repeat("héllo wörld 世界 ", 30)
	chunks, err := Chunk(text, ChunkOptions{Size: 40, Overlap: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range chunks {
		if !strings_ValidUTF8(c) {
			t.Fatalf("chunk is not valid UTF-8: %q", c)
		}
	}
}

func lastNRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[len(r)-n:])
}

func strings_ValidUTF8(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}
