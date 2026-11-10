.PHONY: test test-go test-py bench up down dashboard
test: test-go test-py
test-go:
	go vet ./... && go test -race -count=1 ./router/... && go test -count=1 -short ./benchmark/...
test-py:
	cd engine && python -m pytest -q
bench:
	ARCTIC_WRITE_RESULTS=1 go test ./benchmark -run Headline -v -timeout 600s
up:
	docker compose up --build
down:
	docker compose down
dashboard:
	python deploy/grafana/make_dashboard.py
