VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -ldflags "-X main.version=$(VERSION)"

.PHONY: build test race vet fmt docker clean

build:
	go build -buildvcs=false $(LDFLAGS) -o bin/viiwork-freetoken ./cmd/viiwork-freetoken

test:
	go test ./...

# The mesh is concurrent by nature — peer polling, the health loop and the
# request path all touch backend state — so the race detector is the useful
# gate, not an optional extra.
race:
	go test -race ./...

vet:
	go vet ./...

fmt:
	gofmt -l -w .

docker:
	docker build --build-arg VERSION=$(VERSION) -t viiwork-freetoken:$(VERSION) -t viiwork-freetoken:latest .

clean:
	rm -rf bin
