VERSION ?= 0.1.0

.PHONY: build-windows build-linux verify test clean

build-windows:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 \
	go build -ldflags="-s -w -X main.version=$(VERSION)" \
	-o dist/kaiwu.exe ./cmd/kaiwu/

build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
	go build -ldflags="-s -w -X main.version=$(VERSION)" \
	-o dist/kaiwu ./cmd/kaiwu/

verify:
	GOOS=linux go build ./...
	GOOS=windows go build ./...

test:
	go test ./...

clean:
	rm -rf dist/
