//go:build corpus

package rag

import (
	"context"
	"encoding/json"
	"github.com/ipiton/agent-memory-mcp/internal/config"
	"github.com/ipiton/agent-memory-mcp/internal/vectorstore"
	"os"
	"testing"
)

// TestRankingDiagnostics is opt-in: its caller owns the isolated model and index.
func TestRankingDiagnostics(t *testing.T) {
	configPath := os.Getenv("RAG_DIAGNOSTIC_CONFIG")
	if configPath == "" {
		t.Skip("requires an isolated live-model test configuration")
	}
	cfg, err := config.LoadFromFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	engine := NewEngine(cfg, nil)
	if engine == nil {
		t.Fatal("NewEngine returned nil")
	}
	t.Cleanup(engine.Stop)
	input, err := os.ReadFile(os.Getenv("RAG_DIAGNOSTIC_INPUT"))
	if err != nil {
		t.Fatal(err)
	}
	var queries []struct {
		ID              string   `json:"id"`
		Language        string   `json:"language"`
		Query           string   `json:"query"`
		AcceptedSources []string `json:"accepted_sources"`
		Kind            string   `json:"kind"`
	}
	if err := json.Unmarshal(input, &queries); err != nil {
		t.Fatal(err)
	}
	output, err := os.Create(os.Getenv("RAG_DIAGNOSTIC_OUTPUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	encoder := json.NewEncoder(output)
	store := engine.vecService.store
	ctx := context.Background()
	for _, query := range queries {
		embedding, err := engine.vecService.config.Embedder.EmbedQueryDetailed(ctx, query.Query)
		if err != nil {
			t.Fatal(err)
		}
		allSemantic, err := store.Search(embedding.Embedding, store.Count())
		if err != nil {
			t.Fatal(err)
		}
		semantic := allSemantic[:min(50, len(allSemantic))]
		keyword, err := store.KeywordSearch(query.Query, 100)
		if err != nil {
			t.Fatal(err)
		}
		profiles := make(map[string]any)
		for _, mode := range []string{"weighted", "rrf", "rrf-boosted", "semantic"} {
			rows, _, _ := buildHybridSearchResults(query.Query, "", semantic, keyword, store.Count(), store.Count(), true, newFusionSettings(mode, 60), 0)
			profiles[mode] = diagnosticDistinct(rows, 10)
		}
		full, _, _ := buildHybridSearchResults(query.Query, "", allSemantic, keyword, store.Count(), store.Count(), true, newFusionSettings("weighted", 60), 0)
		profiles["weighted-full-semantic"] = diagnosticDistinct(full, 10)
		withoutDates := func(input []vectorstore.SearchResult) []vectorstore.SearchResult {
			copied := append([]vectorstore.SearchResult(nil), input...)
			for i := range copied {
				copied[i].LastModified = copied[i].LastModified.AddDate(-100, 0, 0)
			}
			return copied
		}
		noRecency, _, _ := buildHybridSearchResults(query.Query, "", withoutDates(semantic), withoutDates(keyword), store.Count(), store.Count(), true, newFusionSettings("weighted", 60), 0)
		profiles["weighted-no-recency"] = diagnosticDistinct(noRecency, 10)
		vectorRows := make([]SearchResult, 0, len(allSemantic))
		for _, result := range allSemantic {
			vectorRows = append(vectorRows, SearchResult{ID: result.ID, Path: result.DocPath, Score: result.Score})
		}
		profiles["vector-only-diagnostic"] = diagnosticDistinct(vectorRows, 10)
		targets := make(map[string]any)
		for _, source := range query.AcceptedSources {
			for i, result := range allSemantic {
				if result.DocPath != source {
					continue
				}
				entry := map[string]any{"best_chunk": result.ID, "semantic_chunk_rank": i + 1, "semantic_score": result.Score, "inside_native_semantic_pool": i < 50}
				for j, row := range full {
					if row.ID == result.ID {
						entry["full_pool_final_chunk_rank"] = j + 1
						entry["full_pool_score"] = row.Debug.Breakdown
						break
					}
				}
				targets[source] = entry
				break
			}
		}
		actual, err := engine.Search(ctx, query.Query, 5, "", true)
		if err != nil {
			t.Fatal(err)
		}
		engine.vecService.config.Fusion = "semantic"
		semanticActual, err := engine.Search(ctx, query.Query, 5, "", true)
		engine.vecService.config.Fusion = cfg.RAG.Fusion
		if err != nil {
			t.Fatal(err)
		}
		profiles["semantic-native"] = semanticActual.Results
		if len(query.AcceptedSources) > 0 {
			found := false
			for _, hit := range semanticActual.Results {
				for _, source := range query.AcceptedSources {
					found = found || hit.Path == source
				}
			}
			if !found {
				t.Errorf("semantic source miss: %s/%s", query.ID, query.Language)
			}
		}
		chunks := make(map[string]string)
		allHits := append([]SearchResult(nil), actual.Results...)
		allHits = append(allHits, semanticActual.Results...)
		for _, hit := range allHits {
			for _, chunk := range allSemantic {
				if chunk.ID == hit.ID {
					chunks[hit.ID] = chunk.Content
					break
				}
			}
		}
		if err := encoder.Encode(map[string]any{"id": query.ID, "language": query.Language, "query": query.Query, "kind": query.Kind, "accepted_sources": query.AcceptedSources, "embedding_provider": embedding.Provider, "query_embedding": embedding.Embedding, "profiles": profiles, "targets": targets, "native": actual, "semantic_native": semanticActual, "chunks": chunks}); err != nil {
			t.Fatal(err)
		}
	}
}

func diagnosticDistinct(rows []SearchResult, limit int) []SearchResult {
	seen := make(map[string]bool)
	var result []SearchResult
	for _, row := range rows {
		if seen[row.Path] {
			continue
		}
		seen[row.Path] = true
		result = append(result, row)
		if len(result) == limit {
			break
		}
	}
	return result
}
