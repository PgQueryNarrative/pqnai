//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/PgQueryNarrative/pqnai/internal/rag"
)

// fakeEmbedder maps known strings to hand-picked vectors so cosine
// distance has a deterministic, checkable winner, without depending on
// a real embedding model being available in CI. The map is shared by
// reference, so subtests can register more fixtures as they go.
type fakeEmbedder struct {
	vectors map[string][]float32
}

func (f fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if v, ok := f.vectors[text]; ok {
		return v, nil
	}
	return make([]float32, 384), nil
}

type fakeGenerator struct {
	answer string // defaults to "fake answer" if unset
	calls  int
}

func (g *fakeGenerator) Generate(ctx context.Context, prompt string) (string, error) {
	g.calls++
	if g.answer == "" {
		return "fake answer", nil
	}
	return g.answer, nil
}

// contents extracts just the text of each source, for tests that don't
// care about citation ids.
func contents(sources []rag.Source) []string {
	out := make([]string, len(sources))
	for i, s := range sources {
		out[i] = s.Content
	}
	return out
}

func vec384(nonZeroIdx int) []float32 {
	v := make([]float32, 384)
	v[nonZeroIdx] = 1
	return v
}

// nearVec384 is vec384(0) nudged by a tiny, distinct amount per i, so many
// fixtures can sit "basically on top of" the same query without being
// exact duplicates.
//
// The nudge is deliberately along dimension 1 -- the direction of vec384(1),
// which testFilteredHNSW uses for the far-away tenant. If these vectors were
// exactly orthogonal to vec384(1), every distance between the two groups
// would be exactly 1.0, HNSW's neighbor-selection heuristic (a strict `<`
// comparison) would prune every edge into the far group on ties, and those
// nodes would be unreachable in the graph under any iterative_scan setting.
// Real embeddings never tie like that, so that fixture would test an
// artifact rather than the actual under-return bug.
func nearVec384(i int) []float32 {
	v := vec384(0)
	v[1] = float32(i) * 1e-4
	return v
}

// TestRAG runs every RAG integration scenario against one live Postgres +
// pgvector container (container startup dominates runtime). Subtests run
// in order and share database state, so later ones scope themselves with
// metadata filters rather than assuming an empty table.
func TestRAG(t *testing.T) {
	ctx := context.Background()

	req := testcontainers.ContainerRequest{
		FromDockerfile: testcontainers.FromDockerfile{
			Context:    "../..",
			Dockerfile: "docker/Dockerfile.postgres",
		},
		ExposedPorts: []string{"5432/tcp"},
		Env: map[string]string{
			"POSTGRES_PASSWORD": "pqnai",
			"POSTGRES_DB":       "pqnai",
		},
		WaitingFor: wait.ForLog("database system is ready to accept connections").WithOccurrence(2),
	}

	pg, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		t.Fatalf("start container: %v", err)
	}
	defer func() { _ = pg.Terminate(ctx) }()

	host, err := pg.Host(ctx)
	if err != nil {
		t.Fatalf("container host: %v", err)
	}
	port, err := pg.MappedPort(ctx, "5432")
	if err != nil {
		t.Fatalf("container port: %v", err)
	}

	dbURL := "postgres://postgres:pqnai@" + host + ":" + port.Port() + "/pqnai"
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	embedder := fakeEmbedder{vectors: map[string][]float32{}}

	t.Run("vector round trip", func(t *testing.T) { testVectorRoundTrip(t, ctx, pool, embedder) })
	t.Run("chunked embed", func(t *testing.T) { testChunkedEmbed(t, ctx, pool, embedder) })
	t.Run("hybrid keyword rescue", func(t *testing.T) { testHybridKeywordRescue(t, ctx, pool, embedder) })
	t.Run("hybrid keyword rescue with natural-language question", func(t *testing.T) { testHybridNaturalQuestion(t, ctx, pool, embedder) })
	t.Run("filters isolate sources", func(t *testing.T) { testFiltersIsolate(t, ctx, pool, embedder) })
	t.Run("stopword-only question falls back to vector", func(t *testing.T) { testStopwordFallback(t, ctx, pool, embedder) })
	t.Run("rerank promotes the relevant chunk", func(t *testing.T) { testRerank(t, ctx, pool, embedder) })
	t.Run("groundedness check", func(t *testing.T) { testGroundedness(t, ctx, pool, embedder) })
	// Must run last: it disables sequential scans database-wide.
	t.Run("filtered hnsw search does not under-return", func(t *testing.T) { testFilteredHNSW(t, ctx, pool, dbURL, embedder) })
}

func testVectorRoundTrip(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	embedder.vectors["about dogs"] = vec384(0)
	embedder.vectors["about cars"] = vec384(1)
	embedder.vectors["query"] = vec384(0)

	for _, text := range []string{"about dogs", "about cars"} {
		if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: text}); err != nil {
			t.Fatalf("embed %q: %v", text, err)
		}
	}

	resp, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{Question: "query", TopK: 1})
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if len(resp.Sources) != 1 || resp.Sources[0].Content != "about dogs" {
		t.Fatalf("expected closest match ['about dogs'], got %q", contents(resp.Sources))
	}
	if resp.Sources[0].ID != 1 {
		t.Fatalf("expected the single source to be cited as id 1, got %d", resp.Sources[0].ID)
	}
}

// testChunkedEmbed proves a long document is actually split into
// multiple rows in pqnai.chunks rather than stored as one oversized
// vector.
func testChunkedEmbed(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	longText := strings.Repeat("sentence about elephants. ", 20) // ~540 runes
	resp, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{
		Text:         longText,
		ChunkSize:    50,
		ChunkOverlap: 5,
	})
	if err != nil {
		t.Fatalf("embed long text: %v", err)
	}
	if resp.ChunkCount <= 1 {
		t.Fatalf("expected multiple chunks for a long document, got %d", resp.ChunkCount)
	}

	var storedCount, maxIndex int
	if err := pool.QueryRow(ctx,
		"SELECT count(*), max(chunk_index) FROM pqnai.chunks WHERE source_id = $1", resp.SourceID,
	).Scan(&storedCount, &maxIndex); err != nil {
		t.Fatalf("inspect chunks: %v", err)
	}
	if storedCount != resp.ChunkCount {
		t.Fatalf("expected %d stored chunk rows, got %d", resp.ChunkCount, storedCount)
	}
	if maxIndex != resp.ChunkCount-1 {
		t.Fatalf("expected chunk_index to run 0..%d, max was %d", resp.ChunkCount-1, maxIndex)
	}
}

var hybridSuite = json.RawMessage(`{"suite": "hybrid"}`)

const (
	hybridTarget   = "The zyzzyva is a genus of tropical weevil."
	hybridQuestion = "what is a zyzzyva"
)

var hybridNoise = []string{
	"Weevils are a large family of beetles.",
	"Beetles are the largest order of insects.",
	"Insects have six legs and three body segments.",
}

// testHybridKeywordRescue is the Phase B accuracy deliverable: a chunk
// that shares an exact rare term with the question but whose embedding
// is far away must be retrieved. The control half proves pure vector
// search really does miss it in this fixture, so the hybrid assertion
// can't pass vacuously.
func testHybridKeywordRescue(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	embedder.vectors[hybridQuestion] = vec384(10)
	embedder.vectors[hybridTarget] = vec384(11) // orthogonal to the question
	for _, n := range hybridNoise {
		embedder.vectors[n] = vec384(10) // identical direction to the question
	}

	for _, text := range append([]string{hybridTarget}, hybridNoise...) {
		if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: text, Metadata: hybridSuite}); err != nil {
			t.Fatalf("embed %q: %v", text, err)
		}
	}

	// Control: vector-only top 3 for this question excludes the target.
	rows, err := pool.Query(ctx, `
		SELECT c.content FROM pqnai.chunks c JOIN pqnai.sources s ON s.id = c.source_id
		WHERE s.metadata @> $1::jsonb
		ORDER BY c.embedding <=> $2::vector LIMIT 3`,
		[]byte(hybridSuite), pgvector.NewVector(vec384(10)).String())
	if err != nil {
		t.Fatalf("control vector query: %v", err)
	}
	var vectorOnly []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			t.Fatalf("scan control row: %v", err)
		}
		vectorOnly = append(vectorOnly, s)
	}
	rows.Close()
	for _, s := range vectorOnly {
		if s == hybridTarget {
			t.Fatalf("control failed: vector-only search already finds the target, so this fixture can't demonstrate hybrid retrieval: %q", vectorOnly)
		}
	}

	// Hybrid: the keyword match is fused in and ranked first.
	resp, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{
		Question: hybridQuestion, TopK: 3, Filters: hybridSuite,
	})
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if len(resp.Sources) == 0 || resp.Sources[0].Content != hybridTarget {
		t.Fatalf("expected hybrid retrieval to rank the keyword match %q first, got %q", hybridTarget, contents(resp.Sources))
	}
}

// testHybridNaturalQuestion guards the AND->OR rewrite in fullTextSearch.
// A real question carries words the target chunk never uses ("get rid
// of"), and plainto_tsquery's AND semantics would then require every one
// of them, so keyword search would match nothing exactly when it's
// needed. The control proves the AND form really misses the target here.
func testHybridNaturalQuestion(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	const question = "how do I get rid of a zyzzyva"
	embedder.vectors[question] = vec384(10) // same direction as the noise, far from the target

	var andMatches int
	if err := pool.QueryRow(ctx, `
		SELECT count(*) FROM pqnai.chunks c JOIN pqnai.sources s ON s.id = c.source_id
		WHERE s.metadata @> $1::jsonb AND c.tsv @@ plainto_tsquery('english', $2)`,
		[]byte(hybridSuite), question,
	).Scan(&andMatches); err != nil {
		t.Fatalf("control AND query: %v", err)
	}
	if andMatches != 0 {
		t.Fatalf("control failed: AND semantics already match %d chunks, so this fixture can't demonstrate the OR rewrite", andMatches)
	}

	resp, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{
		Question: question, TopK: 3, Filters: hybridSuite,
	})
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if len(resp.Sources) == 0 || resp.Sources[0].Content != hybridTarget {
		t.Fatalf("expected keyword match %q first despite extra question words, got %q", hybridTarget, contents(resp.Sources))
	}
}

func testFiltersIsolate(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	// A filter matching nothing returns no sources and never calls the
	// model, rather than letting it answer from an empty context.
	gen := &fakeGenerator{}
	resp, err := rag.Ask(ctx, pool, embedder, gen, nil, nil, rag.AskRequest{
		Question: hybridQuestion, TopK: 3, Filters: json.RawMessage(`{"suite": "does-not-exist"}`),
	})
	if err != nil {
		t.Fatalf("ask with unmatched filter: %v", err)
	}
	if len(resp.Sources) != 0 {
		t.Fatalf("expected no sources for an unmatched filter, got %q", contents(resp.Sources))
	}
	if resp.Answer != rag.NoContextAnswer {
		t.Fatalf("expected %q, got %q", rag.NoContextAnswer, resp.Answer)
	}
	if gen.calls != 0 {
		t.Fatalf("expected the model not to be called with no context, got %d calls", gen.calls)
	}
	if resp.GroundednessError != "no context was retrieved to check" {
		t.Fatalf("expected a no-context explanation, got %q", resp.GroundednessError)
	}

	// A matching filter only ever returns chunks from that suite, even
	// with top_k large enough to include everything in the table.
	allowed := map[string]bool{hybridTarget: true}
	for _, n := range hybridNoise {
		allowed[n] = true
	}
	resp, err = rag.Ask(ctx, pool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{
		Question: hybridQuestion, TopK: 50, Filters: hybridSuite,
	})
	if err != nil {
		t.Fatalf("ask with suite filter: %v", err)
	}
	if len(resp.Sources) != len(allowed) {
		t.Fatalf("expected exactly the %d suite chunks, got %d: %q", len(allowed), len(resp.Sources), contents(resp.Sources))
	}
	for _, s := range resp.Sources {
		if !allowed[s.Content] {
			t.Fatalf("filter leaked a chunk from outside the suite: %q", s.Content)
		}
	}
}

// A question made only of stopwords produces an empty tsquery; full-text
// search must contribute nothing rather than erroring, and retrieval
// must still work via the vector half.
func testStopwordFallback(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	const question = "what is it"
	embedder.vectors[question] = vec384(10)

	resp, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{
		Question: question, TopK: 1, Filters: hybridSuite,
	})
	if err != nil {
		t.Fatalf("ask with stopword-only question: %v", err)
	}
	if len(resp.Sources) != 1 {
		t.Fatalf("expected vector search to still return 1 source, got %q", contents(resp.Sources))
	}
}

// contentReranker scores each passage by a fixed lookup, standing in for
// a model that recognizes which passage actually answers the question.
type contentReranker struct {
	scores map[string]float64
	err    error
}

func (r contentReranker) Score(ctx context.Context, question string, passages []string) ([]float64, error) {
	if r.err != nil {
		return nil, r.err
	}
	out := make([]float64, len(passages))
	for i, p := range passages {
		out[i] = r.scores[p]
	}
	return out, nil
}

// testRerank runs real hybrid retrieval, then re-ranking. The control
// proves retrieval alone puts the answering chunk last, so re-ranking is
// what moves it -- and a failing reranker must leave exactly that
// retrieval order in place rather than failing the question.
func testRerank(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	suite := json.RawMessage(`{"suite": "rerank"}`)
	const question = "which ingredient makes it rise"
	const answer = "Yeast ferments the sugars and produces the gas that leavens bread."
	distractors := []string{
		"Bread is baked in an oven at a high temperature.",
		"Sourdough bread has a tangy flavor.",
		"Bread crust browns through the Maillard reaction.",
	}

	embedder.vectors[question] = vec384(20)
	embedder.vectors[answer] = vec384(21) // far from the question, and no shared lexemes
	for i, d := range distractors {
		// Distinct distances: tied rows have no guaranteed order, and the
		// fallback check below compares two queries' orderings exactly.
		v := vec384(20)
		v[22] = float32(i+1) * 1e-3
		embedder.vectors[d] = v
	}
	for _, text := range append(append([]string{}, distractors...), answer) {
		if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: text, Metadata: suite}); err != nil {
			t.Fatalf("embed %q: %v", text, err)
		}
	}

	base, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{Question: question, TopK: 4, Filters: suite})
	if err != nil {
		t.Fatalf("ask without rerank: %v", err)
	}
	if len(base.Sources) != 4 || base.Sources[3].Content != answer {
		t.Fatalf("control failed: expected retrieval alone to rank the answer last, got %q", contents(base.Sources))
	}
	if base.Reranked {
		t.Fatal("expected reranked=false when re-ranking wasn't requested")
	}

	scores := map[string]float64{answer: 10}
	for _, d := range distractors {
		scores[d] = 1
	}
	reranked, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, contentReranker{scores: scores}, nil,
		rag.AskRequest{Question: question, TopK: 1, Filters: suite, Rerank: true})
	if err != nil {
		t.Fatalf("ask with rerank: %v", err)
	}
	if !reranked.Reranked || reranked.RerankError != "" {
		t.Fatalf("expected a successful rerank, got reranked=%v error=%q", reranked.Reranked, reranked.RerankError)
	}
	if len(reranked.Sources) != 1 || reranked.Sources[0].Content != answer {
		t.Fatalf("expected re-ranking to promote the answer from last to first, got %q", contents(reranked.Sources))
	}

	failed, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{}, contentReranker{err: fmt.Errorf("model unavailable")}, nil,
		rag.AskRequest{Question: question, TopK: 4, Filters: suite, Rerank: true})
	if err != nil {
		t.Fatalf("a reranker failure must not fail the question: %v", err)
	}
	if failed.Reranked || !strings.Contains(failed.RerankError, "model unavailable") {
		t.Fatalf("expected reranked=false with the failure reported, got reranked=%v error=%q", failed.Reranked, failed.RerankError)
	}
	if strings.Join(contents(failed.Sources), "|") != strings.Join(contents(base.Sources), "|") {
		t.Fatalf("expected fallback to the retrieval order %q, got %q", contents(base.Sources), contents(failed.Sources))
	}
}

// fakeChecker is a canned rag.JSONGenerator for the groundedness check:
// its GenerateJSON always returns out, regardless of the prompt.
type fakeChecker struct {
	out string
	err error
}

func (f fakeChecker) GenerateJSON(ctx context.Context, prompt string, schema json.RawMessage) (string, error) {
	return f.out, f.err
}

// testGroundedness exercises CheckGrounded end-to-end through the real
// Ask() pipeline (real retrieval, fake generator/checker): not requested,
// requested and grounded, requested and flagged ungrounded, and requested
// but the checker itself fails -- proving each leaves AskResponse in the
// documented, unambiguous state rather than overlapping with another case.
func testGroundedness(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	suite := json.RawMessage(`{"suite": "groundedness"}`)
	const question = "what color is the sky"
	embedder.vectors[question] = vec384(30)
	embedder.vectors["The sky is blue."] = vec384(30)
	if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: "The sky is blue.", Metadata: suite}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	askArgs := rag.AskRequest{Question: question, TopK: 1, Filters: suite}

	notRequested, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{answer: "The sky is blue."}, nil, nil, askArgs)
	if err != nil {
		t.Fatalf("ask without check_grounded: %v", err)
	}
	if notRequested.Grounded {
		t.Fatal("expected grounded=false when the check wasn't requested")
	}
	if notRequested.GroundednessError != "groundedness check not requested" {
		t.Fatalf("expected an explanation for the unrequested check, got %q", notRequested.GroundednessError)
	}

	askArgs.CheckGrounded = true

	grounded, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{answer: "The sky is blue [1]."}, nil,
		fakeChecker{out: `{"grounded": true, "unsupported": []}`}, askArgs)
	if err != nil {
		t.Fatalf("ask with a grounded verdict: %v", err)
	}
	if !grounded.Grounded || grounded.GroundednessError != "" || len(grounded.Unsupported) != 0 {
		t.Fatalf("expected a clean grounded verdict, got grounded=%v error=%q unsupported=%v",
			grounded.Grounded, grounded.GroundednessError, grounded.Unsupported)
	}

	ungrounded, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{answer: "The sky is blue and it last rained on a Tuesday."}, nil,
		fakeChecker{out: `{"grounded": false, "unsupported": ["it last rained on a Tuesday"]}`}, askArgs)
	if err != nil {
		t.Fatalf("ask with an ungrounded verdict: %v", err)
	}
	if ungrounded.Grounded || ungrounded.GroundednessError != "" {
		t.Fatalf("expected grounded=false with no error (a real verdict, not a failure), got grounded=%v error=%q",
			ungrounded.Grounded, ungrounded.GroundednessError)
	}
	if len(ungrounded.Unsupported) != 1 || ungrounded.Unsupported[0] != "it last rained on a Tuesday" {
		t.Fatalf("expected the unsupported claim to pass through, got %v", ungrounded.Unsupported)
	}

	checkerFailed, err := rag.Ask(ctx, pool, embedder, &fakeGenerator{answer: "The sky is blue."}, nil,
		fakeChecker{err: fmt.Errorf("model unavailable")}, askArgs)
	if err != nil {
		t.Fatalf("a checker failure must not fail the question: %v", err)
	}
	if checkerFailed.Grounded {
		t.Fatal("expected grounded=false when the checker itself failed")
	}
	if !strings.Contains(checkerFailed.GroundednessError, "model unavailable") {
		t.Fatalf("expected the checker failure reported, got %q", checkerFailed.GroundednessError)
	}
	if checkerFailed.Answer == "" {
		t.Fatal("expected the answer to still be returned despite the checker failing")
	}
}

// testFilteredHNSW guards the iterative-scan fix in vectorSearch. With
// pgvector's defaults, an HNSW index scan yields only ef_search (40)
// nearest chunks overall and the metadata filter runs afterwards, so a
// tenant whose chunks all sit outside the global top 40 gets nothing back.
func testFilteredHNSW(t *testing.T, ctx context.Context, pool *pgxpool.Pool, dbURL string, embedder fakeEmbedder) {
	tenantA := json.RawMessage(`{"tenant": "a"}`)
	tenantB := json.RawMessage(`{"tenant": "b"}`)

	// 100 tenant-a chunks crowd the space right around the query...
	for i := 0; i < 100; i++ {
		text := fmt.Sprintf("tenant a document number %d", i)
		embedder.vectors[text] = nearVec384(i)
		if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: text, Metadata: tenantA}); err != nil {
			t.Fatalf("embed tenant a #%d: %v", i, err)
		}
	}
	// ...while tenant b's 3 chunks are far away. None share a lexeme with
	// the question, so full-text search can't mask a vector under-return.
	var tenantBTexts []string
	for i := 0; i < 3; i++ {
		text := fmt.Sprintf("tenant b record %d", i)
		embedder.vectors[text] = vec384(1)
		tenantBTexts = append(tenantBTexts, text)
		if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: text, Metadata: tenantB}); err != nil {
			t.Fatalf("embed tenant b #%d: %v", i, err)
		}
	}

	// Force the HNSW index. With this few rows the planner would otherwise
	// pick an exact plan -- a sequential scan, or filtering sources first
	// via the metadata/source_id indexes and then sorting by distance --
	// and the bug couldn't occur. Disabling sort leaves an ordered HNSW
	// scan as the only cheap way to produce distance order.
	for _, stmt := range []string{
		"ALTER DATABASE pqnai SET enable_seqscan = off",
		"ALTER DATABASE pqnai SET enable_sort = off",
	} {
		if _, err := pool.Exec(ctx, stmt); err != nil {
			t.Fatalf("%s: %v", stmt, err)
		}
	}
	idxPool, err := pgxpool.New(ctx, dbURL) // new connections pick up the setting
	if err != nil {
		t.Fatalf("connect index-forcing pool: %v", err)
	}
	defer idxPool.Close()

	queryVec := pgvector.NewVector(vec384(0)).String()
	const filtered = `
		SELECT c.content FROM pqnai.chunks c JOIN pqnai.sources s ON s.id = c.source_id
		WHERE s.metadata @> $1::jsonb
		ORDER BY c.embedding <=> $2::vector LIMIT 3`

	// Prove the index is actually used, or this whole subtest is vacuous.
	planRows, err := idxPool.Query(ctx, "EXPLAIN "+filtered, []byte(tenantB), queryVec)
	if err != nil {
		t.Fatalf("explain: %v", err)
	}
	var plan strings.Builder
	for planRows.Next() {
		var line string
		if err := planRows.Scan(&line); err != nil {
			t.Fatalf("scan plan: %v", err)
		}
		plan.WriteString(line + "\n")
	}
	planRows.Close()
	if !strings.Contains(plan.String(), "chunks_embedding_idx") {
		t.Fatalf("expected the plan to use the HNSW index, got:\n%s", plan.String())
	}

	// Control: with iterative scan off, this fixture really does under-return.
	tx, err := idxPool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin control tx: %v", err)
	}
	if _, err := tx.Exec(ctx, "SET LOCAL hnsw.iterative_scan = off"); err != nil {
		t.Fatalf("set iterative_scan off: %v", err)
	}
	var controlCount int
	if err := tx.QueryRow(ctx, "SELECT count(*) FROM ("+filtered+") x", []byte(tenantB), queryVec).Scan(&controlCount); err != nil {
		t.Fatalf("control query: %v", err)
	}
	_ = tx.Rollback(ctx)
	if controlCount >= len(tenantBTexts) {
		t.Fatalf("control failed: with iterative_scan off the filtered query still found %d/%d tenant b chunks, so this fixture no longer reproduces the under-return (grow the tenant a set)", controlCount, len(tenantBTexts))
	}

	// The real code path finds all of tenant b's chunks.
	resp, err := rag.Ask(ctx, idxPool, embedder, &fakeGenerator{}, nil, nil, rag.AskRequest{
		Question: "query", TopK: 3, Filters: tenantB,
	})
	if err != nil {
		t.Fatalf("ask as tenant b: %v", err)
	}
	if len(resp.Sources) != len(tenantBTexts) {
		t.Fatalf("expected all %d tenant b chunks (control found only %d), got %d: %q",
			len(tenantBTexts), controlCount, len(resp.Sources), contents(resp.Sources))
	}
	for _, s := range resp.Sources {
		if !strings.HasPrefix(s.Content, "tenant b record") {
			t.Fatalf("tenant b query returned another tenant's chunk: %q", s.Content)
		}
	}
}
