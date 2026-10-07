package rag

import (
	"fmt"
	"strings"
	"unicode"
)

const (
	DefaultChunkSize    = 500
	DefaultChunkOverlap = 50
)

type ChunkOptions struct {
	Size    int
	Overlap int
}

// Chunk splits text into overlapping chunks of roughly Size runes each,
// preferring to break on a paragraph, then sentence, then word boundary
// rather than mid-word, so retrieval doesn't have to match on fragments.
// Operates on runes rather than bytes so multi-byte characters are never
// split across a chunk boundary.
func Chunk(text string, opts ChunkOptions) ([]string, error) {
	if opts.Size <= 0 {
		return nil, fmt.Errorf("rag: chunk size must be positive, got %d", opts.Size)
	}
	if opts.Overlap < 0 {
		return nil, fmt.Errorf("rag: chunk overlap must not be negative, got %d", opts.Overlap)
	}
	if opts.Overlap >= opts.Size {
		return nil, fmt.Errorf("rag: chunk overlap (%d) must be smaller than chunk size (%d)", opts.Overlap, opts.Size)
	}

	runes := []rune(strings.TrimSpace(text))
	if len(runes) == 0 {
		return nil, nil
	}
	if len(runes) <= opts.Size {
		return []string{string(runes)}, nil
	}

	var chunks []string
	start := 0
	for start < len(runes) {
		end := start + opts.Size
		if end >= len(runes) {
			end = len(runes)
		} else {
			end = findBreak(runes, start, end)
		}

		if chunk := strings.TrimSpace(string(runes[start:end])); chunk != "" {
			chunks = append(chunks, chunk)
		}

		if end >= len(runes) {
			break
		}

		next := end - opts.Overlap
		// If the overlap region would straddle a paragraph break, snap
		// forward past it instead: otherwise a chunk that cleanly ended
		// right at a paragraph boundary would have that same boundary
		// pulled back into the middle of the following chunk. Scan a
		// little past `end` too, since a break found by findBreak starts
		// exactly at `end` and wouldn't otherwise be seen here.
		scanTo := end + 2
		if scanTo > len(runes) {
			scanTo = len(runes)
		}
		if j := lastIndexRunes(runes, next, scanTo, "\n\n"); j >= 0 {
			next = j + 2
		}
		if next <= start {
			next = end // guarantee forward progress even in pathological cases
		}
		start = next
	}

	return chunks, nil
}

// findBreak searches backward from `to` (but never past the midpoint of
// [from, to], so a clean break never costs more than half a chunk's worth
// of size) for the best place to end a chunk: a paragraph break, then a
// sentence end, then any whitespace. Falls back to a hard cut at `to`.
func findBreak(runes []rune, from, to int) int {
	minBreak := from + (to-from)/2

	if i := lastIndexRunes(runes, minBreak, to, "\n\n"); i >= 0 {
		return i
	}
	for _, sep := range []string{". ", "! ", "? ", "\n"} {
		if i := lastIndexRunes(runes, minBreak, to, sep); i >= 0 {
			return i + len([]rune(sep))
		}
	}
	for i := to; i > minBreak; i-- {
		if unicode.IsSpace(runes[i-1]) {
			return i
		}
	}
	return to
}

// lastIndexRunes finds the last occurrence of sep within runes[from:to],
// returning its starting index, or -1 if not found.
func lastIndexRunes(runes []rune, from, to int, sep string) int {
	sepRunes := []rune(sep)
	for i := to - len(sepRunes); i >= from; i-- {
		if runesEqual(runes[i:i+len(sepRunes)], sepRunes) {
			return i
		}
	}
	return -1
}

func runesEqual(a, b []rune) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
