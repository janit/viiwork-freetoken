package proxy

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func capturedOutput(t *testing.T, status int, body string) string {
	t.Helper()
	rec := httptest.NewRecorder()
	_, cw := newCaptureWriter(rec)
	if status != 200 {
		cw.WriteHeader(status)
	}
	cw.Write([]byte(body))
	return cw.Output()
}

func TestCaptureNonStreamingJSON(t *testing.T) {
	got := capturedOutput(t, 200, `{"choices":[{"message":{"content":"hello"}}]}`)
	if got != "hello" {
		t.Errorf("got %q", got)
	}
}

func TestCaptureSSEAssemblesTokens(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"a"}}]}

data: {"choices":[{"delta":{"content":"b"}}]}

data: [DONE]

`
	if got := capturedOutput(t, 200, body); got != "ab" {
		t.Errorf("got %q, want ab", got)
	}
}

// A thinking model with reasoning enabled puts everything in
// reasoning_content and leaves content empty. Dropping it would blank the
// output for exactly the requests worth reading.
func TestCaptureKeepsReasoningLabelled(t *testing.T) {
	got := capturedOutput(t, 200, `{"choices":[{"message":{"reasoning_content":"let me think","content":""}}]}`)
	if !strings.Contains(got, "let me think") {
		t.Fatalf("reasoning dropped: %q", got)
	}
	if !strings.Contains(got, "[reasoning]") {
		t.Errorf("reasoning must be labelled, not merged into the answer: %q", got)
	}
}

func TestCaptureSeparatesReasoningFromAnswer(t *testing.T) {
	got := capturedOutput(t, 200, `{"choices":[{"message":{"reasoning_content":"working","content":"result"}}]}`)
	if !strings.Contains(got, "[reasoning]") || !strings.Contains(got, "[answer]") {
		t.Errorf("both parts should be labelled: %q", got)
	}
}

// "What did this request actually do" is exactly the question being asked when
// a request failed; a blank panel answers nothing.
func TestCaptureStoresErrorBody(t *testing.T) {
	got := capturedOutput(t, 503, `{"error":{"message":"no healthy backend"}}`)
	if !strings.Contains(got, "no healthy backend") {
		t.Errorf("error body not stored: %q", got)
	}
}

// The cap exists because SSE runs about 50:1 envelope to text.
func TestCaptureIsBounded(t *testing.T) {
	rec := httptest.NewRecorder()
	_, cw := newCaptureWriter(rec)
	huge := strings.Repeat("x", captureRawCap/2)
	for i := 0; i < 6; i++ {
		cw.Write([]byte(huge))
	}
	if cw.buf.Len() > captureRawCap {
		t.Errorf("captured %d bytes, cap is %d", cw.buf.Len(), captureRawCap)
	}
	// The client still received everything.
	if rec.Body.Len() != len(huge)*6 {
		t.Errorf("client got %d bytes, want %d", rec.Body.Len(), len(huge)*6)
	}
}

func TestExtractPromptFromMessages(t *testing.T) {
	got := extractPromptText([]byte(`{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"why?"}]}`))
	if !strings.Contains(got, "be brief") || !strings.Contains(got, "why?") {
		t.Errorf("got %q", got)
	}
	if !strings.Contains(got, "user:") {
		t.Errorf("roles should be preserved: %q", got)
	}
}

// Multimodal content is an array of typed parts, not a string. The text parts
// are what the store exists to answer with.
func TestExtractPromptFromMultimodalParts(t *testing.T) {
	got := extractPromptText([]byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"describe this"},{"type":"image_url","image_url":{"url":"data:..."}}]}]}`))
	if !strings.Contains(got, "describe this") {
		t.Errorf("text part lost: %q", got)
	}
}

func TestExtractPromptFromCompletionsForm(t *testing.T) {
	if got := extractPromptText([]byte(`{"prompt":"raw completion"}`)); got != "raw completion" {
		t.Errorf("got %q", got)
	}
}

func TestExtractPromptHandlesGarbage(t *testing.T) {
	if got := extractPromptText([]byte(`not json at all`)); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}
