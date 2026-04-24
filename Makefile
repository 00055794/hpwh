.PHONY: build up down logs test smoke

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f app

test:
	python scripts/webapp_local_test.py

smoke:
	python scripts/smoke_local.py
