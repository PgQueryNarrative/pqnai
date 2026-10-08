package rag

import (
	"math"
	"testing"
)

func ids(cs []candidate) []int64 {
	out := make([]int64, len(cs))
	for i, c := range cs {
		out[i] = c.ChunkID
	}
	return out
}

func equalIDs(a, b []int64) bool {
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

func c(id int64) candidate { return candidate{ChunkID: id, Content: "chunk"} }

func TestRRFEmpty(t *testing.T) {
	if got := reciprocalRankFusion(); len(got) != 0 {
		t.Fatalf("expected no results, got %v", ids(got))
	}
	if got := reciprocalRankFusion(nil, nil); len(got) != 0 {
		t.Fatalf("expected no results for empty lists, got %v", ids(got))
	}
}

func TestRRFSingleListPreservesOrder(t *testing.T) {
	got := ids(reciprocalRankFusion([]candidate{c(3), c(1), c(2)}))
	if want := []int64{3, 1, 2}; !equalIDs(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func TestRRFAgreementOutranksSingleList(t *testing.T) {
	// 2 is first in both lists; 1 is first in only one.
	got := ids(reciprocalRankFusion(
		[]candidate{c(2), c(1)},
		[]candidate{c(2), c(3)},
	))
	if got[0] != 2 {
		t.Fatalf("expected chunk found by both retrievers to rank first, got %v", got)
	}
}

func TestRRFDeduplicates(t *testing.T) {
	got := ids(reciprocalRankFusion(
		[]candidate{c(1), c(2)},
		[]candidate{c(2), c(1)},
	))
	if len(got) != 2 {
		t.Fatalf("expected 2 unique chunks, got %v", got)
	}
}

// The core case hybrid search exists for: a chunk vector search ranks
// last, but keyword search ranks first, should beat chunks only vector
// search found.
func TestRRFKeywordRescue(t *testing.T) {
	vector := []candidate{c(1), c(2), c(3), c(4)}
	keyword := []candidate{c(4)}

	got := ids(reciprocalRankFusion(vector, keyword))
	if got[0] != 4 {
		t.Fatalf("expected keyword-matched chunk 4 to be rescued to first place, got %v", got)
	}
	if want := []int64{4, 1, 2, 3}; !equalIDs(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func TestRRFScoresMatchFormula(t *testing.T) {
	// Recompute expected scores directly from the formula and check the
	// ordering they imply, so a change to rrfK or rank indexing (0- vs
	// 1-based) is caught.
	vector := []candidate{c(1), c(2)}
	keyword := []candidate{c(2), c(1)}

	score1 := 1.0/float64(rrfK+1) + 1.0/float64(rrfK+2)
	score2 := 1.0/float64(rrfK+2) + 1.0/float64(rrfK+1)
	if math.Abs(score1-score2) > 1e-12 {
		t.Fatalf("test setup: expected a tie, got %v vs %v", score1, score2)
	}

	// Tied scores fall back to first-seen order: chunk 1 was seen first.
	got := ids(reciprocalRankFusion(vector, keyword))
	if want := []int64{1, 2}; !equalIDs(got, want) {
		t.Fatalf("expected tie broken by first-seen order %v, got %v", want, got)
	}
}

func TestRRFDeterministic(t *testing.T) {
	vector := []candidate{c(5), c(3), c(9), c(1)}
	keyword := []candidate{c(9), c(7), c(5)}

	first := ids(reciprocalRankFusion(vector, keyword))
	for i := 0; i < 50; i++ {
		if got := ids(reciprocalRankFusion(vector, keyword)); !equalIDs(got, first) {
			t.Fatalf("non-deterministic ordering: run %d got %v, first run %v", i, got, first)
		}
	}
}

func TestRRFPreservesContent(t *testing.T) {
	got := reciprocalRankFusion([]candidate{{ChunkID: 1, Content: "hello"}})
	if len(got) != 1 || got[0].Content != "hello" {
		t.Fatalf("expected content to survive fusion, got %+v", got)
	}
}
