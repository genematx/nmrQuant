# Declare constants
IMAGE_NAME = "qt_nmr"
PWD = $(shell pwd)
TEST_PATH = tests

#- Development and Debugging
## Build the development Docker image
build: 
	docker build \
	-t $(IMAGE_NAME):dev -f docker/Dockerfile --target=dev .

## Open a bash shell inside the local dev container
bash: build
	docker run -it --rm \
	-e CI \
	-v $(PWD):/opt \
	$(IMAGE_NAME):dev bash 

#- Testing and formatting
## Run unit tests locally in Docker
test: build
	docker run --rm \
	-e CI \
	-v $(PWD):/opt \
	$(IMAGE_NAME):dev \
	test $(if $(test-args),$(test-args),$(TEST_PATH))

## Run style checks
lint: build
	docker run --rm \
	-e CI \
	-v $(PWD):/opt \
	$(IMAGE_NAME):dev \
	lint

## Run code formatters
fmt: build
	docker run --rm \
	-e CI \
	-v $(PWD):/opt \
	$(IMAGE_NAME):dev \
	fmt $(isort-args) $(black-args)

## Build the production Docker image
build-prod: 
	docker build \
	-t $(IMAGE_NAME):latest -f docker/Dockerfile --target=prod .

## Launch a Jupyter notebook server
notebook: build
	docker run -it \
	--env-file .env \
	-v $(PWD):/opt \
	-p 8888:8888 \
	$(IMAGE_NAME):dev jupyter notebook --allow-root --ip=0.0.0.0 --port=8888 --notebook-dir=/opt

## Prune Docker containers, networks, and images
clean:
	docker system prune -f
