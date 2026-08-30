module github.com/janit/viiwork-freetoken

go 1.27.0

require (
	// The mesh wire contract. viiwork is the source of truth for the protocol and
	// publishes it as a public module; this project is a second implementation of
	// the same contract, driving vLLM on CUDA hardware.
	//
	// There is deliberately NO `replace` directive here. This file is published
	// verbatim to the public repo, and a replace pointing at a private checkout
	// would make the public module unbuildable for everyone — including you, on
	// any machine that does not happen to have viiwork-private beside it. Local
	// development against an unpublished meshapi change uses go.work instead,
	// which is gitignored and never published.
	//
	// This version must name a PUBLISHED viiwork tag, because Go resolves it to
	// build the module graph even when the workspace overrides the source. Bump it
	// to the tag that actually contains the meshapi change you depend on before
	// publishing; scripts/publish.sh builds with GOWORK=off and refuses to publish
	// if the public form does not compile, which is precisely this mistake.
	github.com/janit/viiwork v1.6.2
	gopkg.in/yaml.v3 v3.0.1
)
