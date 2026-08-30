// Package model holds the OpenAI-compatible model list types, shared by the
// peer registry that aggregates them and the proxy that serves them.
package model

import "strings"

// ModelEntry is one entry in an OpenAI /v1/models response.
type ModelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

type ModelsResponse struct {
	Object string       `json:"object"`
	Data   []ModelEntry `json:"data"`
}

// IDFromPath derives a model id from a filesystem path or HuggingFace repo id,
// used only when a config omits an explicit name. "org/Qwen3-32B-AWQ" and
// "/models/Qwen3-32B-AWQ" both yield "Qwen3-32B-AWQ".
func IDFromPath(p string) string {
	p = strings.TrimSuffix(strings.TrimSpace(p), "/")
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		p = p[i+1:]
	}
	return p
}
