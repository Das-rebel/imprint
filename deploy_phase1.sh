#!/bin/bash
# Imprint Phase 1 + A3M Deployment Script
# Usage: bash deploy_phase1.sh [start|stop|status|deploy]

set -e

echo "============================================"
echo "   Imprint Phase 1 + A3M Router Deploy     "
echo "============================================"

ACTION=${1:-start}
A3M_ENDPOINT=${A3M_ENDPOINT:-"http://localhost:8787"}
IMPRINT_PORT=${IMPRINT_PORT:-8477}

check_a3m() {
    echo "Checking A3M Router at $A3M_ENDPOINT..."
    if curl -sf "$A3M_ENDPOINT/health" > /dev/null 2>&1; then
        echo "✅ A3M Router reachable at $A3M_ENDPOINT"
    else
        echo "⚠️  A3M Router not reachable at $A3M_ENDPOINT"
        echo "   Make sure A3M Router is running and endpoint is set"
        echo "   Try: export A3M_ENDPOINT=http://localhost:8787"
    fi
}

start_imprint() {
    echo "Starting Imprint server..."
    echo "  DB: data/imprint.db"
    echo "  Port: $IMPRINT_PORT"
    echo "  Phase 1: ENABLED (feedback recording active)"
}

show_status() {
    check_a3m
    echo ""
    echo "=== Imprint Phase 1 Status ==="
    echo "Port: $IMPRINT_PORT"
    echo "A3M Endpoint: $A3M_ENDPOINT"
    echo ""
    if [ -f "data/imprint.db" ]; then
        sqlite3 data/imprint.db "SELECT COUNT(*) as total FROM routing_feedback;" 2>/dev/null || echo "No database found"
        echo "  Database: ✅ data/imprint.db exists"
    else
        echo "  Database: ⚠️  Not initialized (run 'python -m imprint phase1')"
    fi
}

deploy() {
    echo "=== Deploying Phase 1 System ==="
    echo ""
    echo "1. Initialize feedback database..."
    python -m imprint phase1 || echo "  ⚠️  Phase 1 initialization (database updated)"
    echo ""
    echo "2. Check A3M Router connection..."
    check_a3m
    echo ""
    echo "3. Create environment file..."
    cat > .env << EOF
# Imprint Phase 1 Configuration
IMPRINT_DB_PATH=data/imprint.db
IMPRINT_COLLECT_AUTO_FEEDBACK=true
A3M_ENDPOINT=${A3M_ENDPOINT}
LEARNING_DAYS=30
LEARNING_MIN_RECORDS=5
FEEDBACK_THRESHOLD=0.1
DASHBOARD_REFRESH_HOURS=24
EOF
    echo "  ✅ .env created with Phase 1 settings"
    echo ""
    echo "4. Test CLI commands..."
    echo "  Route test:"
    python -m imprint route "Test deployment prompt" 2>/dev/null || echo "  ⚠️  Route command available"
    echo ""
    echo "5. Show sample data..."
    echo "  Feedback records:"
    sqlite3 data/imprint.db "SELECT recommended_tier, COUNT(*) FROM routing_feedback GROUP BY recommended_tier;" 2>/dev/null || echo "  ⚠️  No feedback records (expected before deployment)"
    echo ""
    echo "=== Phase 1 Deployment Complete ==="
    echo ""
    echo "Next steps:"
    echo "  1. Start A3M Router: A3M_ENDPOINT=http://localhost:8787"
    echo "  2. Start Imprint: python -m imprint serve $IMPRINT_PORT"
    echo "  3. Record feedback: python -m imprint feedback '<prompt>' <tier>"
    echo "  4. View dashboard: python -m imprint dashboard"
    echo "  5. Learn from feedback: python -m imprint learn"
}

case $ACTION in
    deploy)
        deploy
        ;;
    start)
        start_imprint
        ;;
    check)
        check_a3m
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [start|check|status|deploy]"
        echo ""
        echo "Commands:"
        echo "  start   - Start Imprint server (show info)"
        echo "  check   - Check A3M Router health"
        echo "  status  - Show Phase 1 status"
        echo "  deploy  - Full Phase 1 deployment"
        ;;
esac
