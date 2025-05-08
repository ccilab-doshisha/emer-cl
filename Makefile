NAME=emer-cl
DOCKERFILE_PATH=$(PWD)/docker
HOST_WORK=$(PWD)
CONTAINER_WORK=/work
CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

# Build
build: build-cpu
build-cpu:
	docker build --build-arg docker_tag=1.15.5-py3 -t $(NAME) -f $(DOCKERFILE_PATH)/Dockerfile $(DOCKERFILE_PATH)
build-gpu:
	docker build --build-arg docker_tag=1.15.5-gpu -t $(NAME) -f $(DOCKERFILE_PATH)/Dockerfile $(DOCKERFILE_PATH)

# Create tensorflow container for EMER-CL
run:
	docker run -itd --name $(NAME) -v $(HOST_WORK):$(CONTAINER_WORK) -u $(CURRENT_UID):$(CURRENT_GID) -w $(CONTAINER_WORK) $(NAME)
run-gpu:
	docker run -itd --gpus all --name $(NAME) -v $(HOST_WORK):$(CONTAINER_WORK) -u $(CURRENT_UID):$(CURRENT_GID) -w $(CONTAINER_WORK) $(NAME)

# Attach contaier
attach:
	docker exec -it $(NAME) /bin/bash

# Shortcut
up: up-cpu
up-cpu: build run
up-gpu: build-gpu run-gpu

down: stop rm

# Restart container is stopped 
start:
	docker start $(NAME)

# Stop container
stop:
	docker stop $(NAME)

# Remove container
rm:
	docker rm $(NAME)

log:
	docker logs $(NAME) --follow
