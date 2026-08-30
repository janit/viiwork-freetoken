package freetoken

import (
	"bytes"
	"io"
	"sync"
)

// prefixWriter tags each line of a subprocess's output with the backend it
// came from, so one log stream from a multi-backend node stays readable.
//
// It buffers partial lines rather than prefixing every Write: a subprocess
// writing a line in several syscalls would otherwise get a prefix mid-line.
type prefixWriter struct {
	mu     sync.Mutex
	out    io.Writer
	prefix []byte
	buf    bytes.Buffer
}

func newPrefixWriter(out io.Writer, prefix string) *prefixWriter {
	return &prefixWriter{out: out, prefix: []byte(prefix)}
}

func (w *prefixWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(p)
	w.buf.Write(p)
	for {
		line, err := w.buf.ReadBytes('\n')
		if err != nil {
			// Partial line: keep it for the next write rather than emitting a
			// prefix that will end up mid-sentence.
			w.buf.Write(line)
			break
		}
		if _, err := w.out.Write(append(append([]byte(nil), w.prefix...), line...)); err != nil {
			return n, err
		}
	}
	return n, nil
}
