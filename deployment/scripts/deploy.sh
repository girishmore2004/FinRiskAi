deployment_script = """
#!/bin/bash

# FinRiskAI+ Deployment Script
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-staging}
PROJECT_NAME="finriskai"
DOCKER_REGISTRY="your-registry.com"

echo "🚀 Starting deployment for environment: $ENVIRONMENT"

# Check if required tools are installed
check_requirements() {
    command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
}

# Build Docker images
build_images() {
    echo "📦 Building Docker images..."
    
    # Main application
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-app:latest -f deployment/docker/Dockerfile .
    
    # Model trainer
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-trainer:latest -f deployment/docker/Dockerfile.trainer .
    
    # Web dashboard
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-dashboard:latest -f deployment/docker/Dockerfile.dashboard .
}

# Push images to registry
push_images() {
    echo "📤 Pushing images to registry..."
    docker push $DOCKER_REGISTRY/$PROJECT_NAME-app:latest
    docker push $DOCKER_REGISTRY/$PROJECT_NAME-trainer:latest
    docker push $DOCKER_REGISTRY/$PROJECT_NAME-dashboard:latest
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo "☸️ Deploying to Kubernetes..."
    
    # Create namespace if not exists
    kubectl create namespace $PROJECT_NAME --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f deployment/kubernetes/configmap.yaml -n $PROJECT_NAME
    kubectl apply -f deployment/kubernetes/secrets.yaml -n $PROJECT_NAME
    kubectl apply -f deployment/kubernetes/deployment.yaml -n $PROJECT_NAME
    kubectl apply -f deployment/kubernetes/service.yaml -n $PROJECT_NAME
    kubectl apply -f deployment/kubernetes/ingress.yaml -n $PROJECT_NAME
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/$PROJECT_NAME-app -n $PROJECT_NAME
}

# Run database migrations
run_migrations() {
    echo "🗄️ Running database migrations..."
    kubectl run migration-job --image=$DOCKER_REGISTRY/$PROJECT_NAME-app:latest \
        --restart=Never --rm -i --tty \
        --namespace=$PROJECT_NAME \
        --command -- python -m alembic upgrade head
}

# Health check
health_check() {
    echo "🏥 Performing health check..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service $PROJECT_NAME-service -n $PROJECT_NAME -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -n "$SERVICE_IP" ]; then
        echo "Service available at: http://$SERVICE_IP"
        
        # Wait for service to be ready
        for i in {1..30}; do
            if curl -f http://$SERVICE_IP/health > /dev/null 2>&1; then
                echo "✅ Health check passed!"
                break
            fi
            echo "Waiting for service to be ready... ($i/30)"
            sleep 10
        done
    else
        echo "❌ Could not determine service IP"
        exit 1
    fi
}

# Main deployment flow
main() {
    check_requirements
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "⚠️ Deploying to PRODUCTION environment"
        read -p "Are you sure you want to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    build_images
    push_images
    deploy_kubernetes
    run_migrations
    health_check
    
    echo "🎉 Deployment completed successfully!"
    echo "🔗 Access the application at: https://finriskai.your-domain.com"
}

# Execute main function
main "$@"
"""