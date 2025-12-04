#!/bin/bash

# Environment Switching Script
# Usage: ./switch-env.sh [local|staging]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}Usage:${NC} ./switch-env.sh [local|staging|check]"
    echo ""
    echo "Commands:"
    echo "  local    - Switch to local development environment"
    echo "  staging  - Switch to staging environment"
    echo "  check    - Check current environment configuration"
    echo ""
    exit 1
}

check_env() {
    echo -e "${BLUE}📋 Current Environment Configuration:${NC}"
    echo ""
    
    if [ ! -f ".env" ]; then
        echo -e "${RED}❌ No .env file found!${NC}"
        echo "Run './switch-env.sh local' or './switch-env.sh staging' first"
        return 1
    fi
    
    CURRENT_ENV=$(grep "^ENVIRONMENT=" .env | cut -d= -f2)
    CURRENT_BUCKET=$(grep "^S3_BUCKET_NAME=" .env | cut -d= -f2)
    CURRENT_QUEUE=$(grep "^SQS_QUEUE_URL=" .env | cut -d= -f2)
    
    echo -e "Environment: ${GREEN}$CURRENT_ENV${NC}"
    echo -e "S3 Bucket:   ${GREEN}$CURRENT_BUCKET${NC}"
    echo -e "SQS Queue:   ${GREEN}$CURRENT_QUEUE${NC}"
    echo ""
    
    if [ "$CURRENT_ENV" == "local" ]; then
        echo -e "${YELLOW}ℹ️  Local Environment:${NC}"
        echo "   - Uses rocabucket-1"
        echo "   - Uses doc-image-queue"
        echo "   - No Lambda processing"
        echo "   - Backend handles all processing"
    elif [ "$CURRENT_ENV" == "staging" ]; then
        echo -e "${YELLOW}ℹ️  Staging Environment:${NC}"
        echo "   - Uses staging bucket"
        echo "   - Uses staging queues"
        echo "   - Lambda processes files"
        echo "   - Pre-production testing"
    fi
}

switch_local() {
    echo -e "${BLUE}🔄 Switching to LOCAL environment...${NC}"
    
    if [ ! -f ".env.local" ]; then
        echo -e "${YELLOW}⚠️  .env.local not found, creating template...${NC}"
        cat > .env.local << 'EOF'
# Local Development Environment
ENVIRONMENT=local

# AWS Configuration
AWS_REGION=me-central-1
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY

# Local AWS Resources
S3_BUCKET_NAME=rocabucket-1
SQS_QUEUE_URL=https://sqs.me-central-1.amazonaws.com/930816733230/doc-image-queue
PROCESSED_QUEUE_URL=https://sqs.me-central-1.amazonaws.com/930816733230/doc-processed-queue

# Anthropic API
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
EOF
        echo -e "${RED}❌ Please edit .env.local with your actual credentials!${NC}"
        return 1
    fi
    
    cp .env.local .env
    echo -e "${GREEN}✅ Switched to LOCAL environment${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "   - S3: rocabucket-1"
    echo "   - SQS: doc-image-queue"
    echo "   - Lambda: DISABLED"
    echo "   - Backend: Handles all processing"
    echo ""
    echo -e "${YELLOW}💡 Start backend with:${NC} python main.py"
}

switch_staging() {
    echo -e "${BLUE}🔄 Switching to STAGING environment...${NC}"
    
    if [ ! -f ".env.staging" ]; then
        echo -e "${YELLOW}⚠️  .env.staging not found, creating template...${NC}"
        cat > .env.staging << 'EOF'
# Staging Environment
ENVIRONMENT=staging

# AWS Configuration
AWS_REGION=me-central-1
# Note: In staging deployment, use IAM role instead of keys
# Only set these for local testing of staging config
# AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
# AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY

# Staging AWS Resources
S3_BUCKET_NAME=rocabucket-staging
SQS_QUEUE_URL=https://sqs.me-central-1.amazonaws.com/930816733230/staging-doc-image-queue
PROCESSED_QUEUE_URL=https://sqs.me-central-1.amazonaws.com/930816733230/staging-doc-processed-queue

# Anthropic API
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
EOF
        echo -e "${RED}❌ Please edit .env.staging with your actual staging configuration!${NC}"
        return 1
    fi
    
    cp .env.staging .env
    echo -e "${GREEN}✅ Switched to STAGING environment${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "   - S3: rocabucket-staging (or your staging bucket)"
    echo "   - SQS: staging-doc-image-queue"
    echo "   - Lambda: ENABLED"
    echo "   - Backend: Sends to queue, Lambda processes"
    echo ""
    echo -e "${YELLOW}💡 Start backend with:${NC} python main.py"
}

# Main script
if [ $# -eq 0 ]; then
    print_usage
fi

case "$1" in
    local)
        switch_local
        echo ""
        check_env
        ;;
    staging)
        switch_staging
        echo ""
        check_env
        ;;
    check)
        check_env
        ;;
    *)
        echo -e "${RED}❌ Invalid command: $1${NC}"
        echo ""
        print_usage
        ;;
esac

