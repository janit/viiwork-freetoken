package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
)

// captureRawCap bounds what one response may contribute to the prompt store.
//
// 2 MB sounds generous only until you account for SSE, which runs about 50:1
// envelope to text: every token arrives wrapped in a JSON chunk with its own
// role, index and finish_reason. The cap is on the raw bytes because that is
// what we accumulate; the extracted text is capped again, far lower, by the
// store itself.
const captureRawCap = 2 << 20

// captureWriter tees the response body while it is being written to the
// client, so the prompt store records what the client actually received.
//
// Bytes are appended raw and parsed once, after the response completes.
// Decoding each SSE chunk on the way past would put a JSON parse on the
// per-token path; an append into a byte slice is a memcpy instead.
type captureWriter struct {
	http.ResponseWriter
	buf      bytes.Buffer
	status   int
	overflow bool
}

func newCaptureWriter(w http.ResponseWriter) (http.ResponseWriter, *captureWriter) {
	cw := &captureWriter{ResponseWriter: w, status: http.StatusOK}
	return cw, cw
}

func (c *captureWriter) WriteHeader(status int) {
	c.status = status
	c.ResponseWriter.WriteHeader(status)
}

func (c *captureWriter) Write(p []byte) (int, error) {
	if c.buf.Len()+len(p) <= captureRawCap {
		c.buf.Write(p)
	} else {
		c.overflow = true
	}
	return c.ResponseWriter.Write(p)
}

// Flush forwards to the wrapped writer. Without this the wrapper silently
// swallows http.Flusher and every streamed response buffers until completion —
// which turns token-by-token output into one late block.
func (c *captureWriter) Flush() {
	if f, ok := c.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Output returns the assistant text from the captured response.
//
// A failed request stores its error body instead, because "what did this
// request actually do" is exactly the question being asked when a request
// failed, and a blank panel answers nothing.
func (c *captureWriter) Output() string {
	body := c.buf.Bytes()
	if len(body) == 0 {
		return ""
	}
	if c.status >= 400 {
		return string(body)
	}
	if bytes.Contains(body, []byte("data: ")) {
		return extractSSEText(body)
	}
	return extractJSONText(body)
}

// chunk is the subset of an OpenAI-compatible response this needs. Reasoning
// is kept separate rather than merged: a thinking model with reasoning enabled
// puts everything in reasoning_content and leaves content empty, so dropping
// it would blank the output for exactly the requests worth reading.
type chunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
		Text string `json:"text"`
	} `json:"choices"`
}

func extractSSEText(body []byte) string {
	var content, reasoning strings.Builder
	for _, line := range bytes.Split(body, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := bytes.TrimPrefix(line, []byte("data: "))
		if bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var c chunk
		if err := json.Unmarshal(payload, &c); err != nil {
			continue
		}
		for _, ch := range c.Choices {
			content.WriteString(ch.Delta.Content)
			content.WriteString(ch.Message.Content)
			content.WriteString(ch.Text)
			reasoning.WriteString(ch.Delta.ReasoningContent)
			reasoning.WriteString(ch.Message.ReasoningContent)
		}
	}
	return joinReasoning(reasoning.String(), content.String())
}

func extractJSONText(body []byte) string {
	var c chunk
	if err := json.Unmarshal(body, &c); err != nil {
		return string(body)
	}
	var content, reasoning strings.Builder
	for _, ch := range c.Choices {
		content.WriteString(ch.Message.Content)
		content.WriteString(ch.Delta.Content)
		content.WriteString(ch.Text)
		reasoning.WriteString(ch.Message.ReasoningContent)
		reasoning.WriteString(ch.Delta.ReasoningContent)
	}
	if content.Len() == 0 && reasoning.Len() == 0 {
		return string(body)
	}
	return joinReasoning(reasoning.String(), content.String())
}

// joinReasoning labels reasoning rather than merging it into the answer, so a
// reader can tell the model's working from its reply.
func joinReasoning(reasoning, content string) string {
	if reasoning == "" {
		return content
	}
	if content == "" {
		return "[reasoning]\n" + reasoning
	}
	return "[reasoning]\n" + reasoning + "\n\n[answer]\n" + content
}

// extractPromptText pulls the user-visible prompt from a request body.
//
// Multimodal content arrives as an array of typed parts rather than a string;
// the text parts are joined and the rest ignored, because the store exists to
// answer "what was asked", and an image is not answerable in a text panel.
func extractPromptText(body []byte) string {
	var req struct {
		Prompt   json.RawMessage `json:"prompt"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	if len(req.Messages) > 0 {
		var sb strings.Builder
		for _, m := range req.Messages {
			text := contentText(m.Content)
			if text == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteString("\n\n")
			}
			sb.WriteString(m.Role)
			sb.WriteString(": ")
			sb.WriteString(text)
		}
		return sb.String()
	}
	if len(req.Prompt) > 0 {
		return contentText(req.Prompt)
	}
	return ""
}

func contentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			if p.Text == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(p.Text)
		}
		return sb.String()
	}
	// A completions-style array of strings.
	var strs []string
	if err := json.Unmarshal(raw, &strs); err == nil {
		return strings.Join(strs, "\n")
	}
	return ""
}
