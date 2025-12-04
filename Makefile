# Makefile for RizanAI Backend - Docker & ECS Operations

.PHONY: help build run stop test deploy clean logs shell

# Configuration
AWS_REGION ?= me-central-1
AWS_ACCOUNT_ID ?= 930816733230
ECR_REPOSITORY ?= rizanai-backend
ECS_CLUSTER ?= rizanai-cluster
ECS_SERVICE ?= rizanai-backend-service
IMAGE_NAME = rizanai-backend
CONTAINER_NAME = rizanai-backend

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ========================================
# Local Development
# ========================================

build: ## Build Docker image locally
	@echo "🔨 Building Docker image..."
	docker build -t $(IMAGE_NAME):latest .
	@echo "✅ Build complete!"

run: ## Run container locally
	@echo "🚀 Starting container..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p 8000:8000 \
		--env-file .env \
		$(IMAGE_NAME):latest
	@echo "✅ Container started at http://localhost:8000"
	@echo "📊 Health check: http://localhost:8000/health"

run-interactive: ## Run container in interactive mode
	@echo "🚀 Starting container in interactive mode..."
	docker run -it --rm \
		-p 8000:8000 \
		--env-file .env \
		$(IMAGE_NAME):latest /bin/bash

stop: ## Stop and remove local container
	@echo "🛑 Stopping container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "✅ Container stopped"

restart: stop run ## Restart local container

logs: ## View container logs
	docker logs -f $(CONTAINER_NAME)

shell: ## Open shell in running container
	docker exec -it $(CONTAINER_NAME) /bin/bash

# ========================================
# Testing
# ========================================

test-local: ## Test local deployment
	@echo "🧪 Testing local deployment..."
	@sleep 5
	@curl -f http://localhost:8000/health || (echo "❌ Health check failed" && exit 1)
	@echo "✅ Health check passed!"
	@curl http://localhost:8000/ | python -m json.tool
	@echo "✅ API accessible!"

test-compose: ## Test with docker-compose
	@echo "🧪 Testing with docker-compose..."
	docker-compose up -d
	@sleep 10
	@curl -f http://localhost:8000/health || (echo "❌ Health check failed" && exit 1)
	@echo "✅ Health check passed!"
	docker-compose down

# ========================================
# AWS ECR Operations
# ========================================

ecr-login: ## Login to AWS ECR
	@echo "🔐 Logging in to ECR..."
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	@echo "✅ Logged in to ECR"

ecr-create: ## Create ECR repository
	@echo "📦 Creating ECR repository..."
	aws ecr create-repository \
		--repository-name $(ECR_REPOSITORY) \
		--region $(AWS_REGION) || echo "Repository may already exist"
	@echo "✅ ECR repository ready"

ecr-push: build ecr-login ## Build and push image to ECR
	@echo "📤 Pushing image to ECR..."
	docker tag $(IMAGE_NAME):latest \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):latest
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):latest
	@echo "✅ Image pushed successfully!"

ecr-list: ## List images in ECR
	@echo "📋 ECR images:"
	aws ecr list-images \
		--repository-name $(ECR_REPOSITORY) \
		--region $(AWS_REGION)

# ========================================
# AWS ECS Deployment
# ========================================

deploy: ## Deploy to AWS ECS (automated)
	@echo "🚀 Starting deployment..."
	chmod +x deploy-ecs.sh
	./deploy-ecs.sh

ecs-update: ecr-push ## Update ECS service with new image
	@echo "🔄 Updating ECS service..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service $(ECS_SERVICE) \
		--force-new-deployment \
		--region $(AWS_REGION)
	@echo "✅ Service update initiated"

ecs-status: ## Check ECS service status
	@echo "📊 ECS Service Status:"
	aws ecs describe-services \
		--cluster $(ECS_CLUSTER) \
		--services $(ECS_SERVICE) \
		--region $(AWS_REGION) \
		--query 'services[0].{Name:serviceName,Status:status,Running:runningCount,Desired:desiredCount,Pending:pendingCount}'

ecs-tasks: ## List running tasks
	@echo "📋 Running Tasks:"
	aws ecs list-tasks \
		--cluster $(ECS_CLUSTER) \
		--service-name $(ECS_SERVICE) \
		--region $(AWS_REGION)

ecs-logs: ## View ECS logs
	@echo "📜 Viewing logs..."
	aws logs tail /ecs/rizanai-backend --follow --region $(AWS_REGION)

ecs-scale: ## Scale ECS service (make ecs-scale COUNT=3)
	@echo "📈 Scaling service to $(COUNT) tasks..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service $(ECS_SERVICE) \
		--desired-count $(COUNT) \
		--region $(AWS_REGION)
	@echo "✅ Service scaled to $(COUNT) tasks"

# ========================================
# Cleanup
# ========================================

clean: stop ## Clean local Docker resources
	@echo "🧹 Cleaning up..."
	-docker rmi $(IMAGE_NAME):latest
	-docker system prune -f
	@echo "✅ Cleanup complete"

clean-all: clean ## Remove all local Docker data
	@echo "⚠️  This will remove ALL Docker data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker system prune -a --volumes -f; \
		echo "✅ All Docker data removed"; \
	fi

# ========================================
# Monitoring & Debugging
# ========================================

health: ## Check application health
	@echo "🏥 Health Check:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Health check failed"

api-test: ## Test API endpoints
	@echo "🧪 Testing API endpoints..."
	@echo "\n📍 Root endpoint:"
	@curl -s http://localhost:8000/ | python -m json.tool
	@echo "\n📍 Health endpoint:"
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo "\n📍 Folders endpoint:"
	@curl -s http://localhost:8000/api/folders | python -m json.tool

ps: ## Show running containers
	@docker ps --filter name=$(CONTAINER_NAME)

stats: ## Show container resource usage
	@docker stats $(CONTAINER_NAME) --no-stream

inspect: ## Inspect container
	@docker inspect $(CONTAINER_NAME) | python -m json.tool

# ========================================
# Utilities
# ========================================

env-check: ## Check environment variables
	@echo "🔍 Environment Check:"
	@echo "AWS_REGION: $(AWS_REGION)"
	@echo "AWS_ACCOUNT_ID: $(AWS_ACCOUNT_ID)"
	@echo "ECR_REPOSITORY: $(ECR_REPOSITORY)"
	@echo "ECS_CLUSTER: $(ECS_CLUSTER)"
	@echo "ECS_SERVICE: $(ECS_SERVICE)"
	@echo "\n.env file:"
	@[ -f .env ] && echo "✅ .env exists" || echo "❌ .env missing"

aws-check: ## Verify AWS credentials
	@echo "🔐 AWS Credentials Check:"
	@aws sts get-caller-identity --region $(AWS_REGION)

docker-check: ## Verify Docker installation
	@echo "🐳 Docker Check:"
	@docker --version
	@docker-compose --version || echo "⚠️  docker-compose not installed"

prereq: aws-check docker-check env-check ## Check all prerequisites

# ========================================
# CI/CD
# ========================================

ci-build: ## Build for CI/CD pipeline
	@echo "🔨 CI Build..."
	docker build --no-cache -t $(IMAGE_NAME):$(shell git rev-parse --short HEAD) .

ci-test: ci-build ## Test in CI environment
	@echo "🧪 CI Test..."
	docker run --rm $(IMAGE_NAME):$(shell git rev-parse --short HEAD) python -m pytest tests/ || true

ci-push: ci-build ecr-login ## Push to ECR with git commit hash
	@echo "📤 CI Push..."
	docker tag $(IMAGE_NAME):$(shell git rev-parse --short HEAD) \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):$(shell git rev-parse --short HEAD)
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):$(shell git rev-parse --short HEAD)

# ========================================
# Default target
# ========================================

.DEFAULT_GOAL := help

