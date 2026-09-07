.PHONY: build up enter down stop logs clean

CONTAINER_NAME = green-challenge

# Set RUN_BENCHMARK=0 to start the container without launching the benchmark.
RUN_BENCHMARK ?= 1

build:
	@echo "Building Docker image..."
	cd docker && docker compose build

up:
	@echo "Starting container..."
	cd docker && docker compose up -d
	@if [ "$(RUN_BENCHMARK)" != "0" ]; then \
		echo "Launching benchmark..."; \
		cd docker && docker compose exec -T -e RESULTS_PATH="$$RESULTS_PATH" -e RUN_PATH="$$RUN_PATH" -e LIVESTREAM="$${LIVESTREAM:-0}" -e PUBLIC_IP="$${PUBLIC_IP:-127.0.0.1}" -e POLICY_PORT="$${POLICY_PORT:-8999}" $(CONTAINER_NAME) bash -lc "cd /workspace/green_challenge && ./run_benchmark.sh"; \
	fi

enter:
	@echo "Entering container..."
	cd docker && docker compose exec $(CONTAINER_NAME) bash

stop:
	@echo "Stopping container..."
	cd docker && docker compose stop

down:
	@echo "Stopping and removing container..."
	cd docker && docker compose down

logs:
	@echo "Showing logs..."
	cd docker && docker compose logs -f $(CONTAINER_NAME)

clean:
	@echo "Cleaning up..."
	cd docker && docker compose down -v
	docker rmi green-challenge:latest 2>/dev/null || true
	@echo "Clean complete"
