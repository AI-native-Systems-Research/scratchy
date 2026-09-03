#!/bin/bash
# Monitor OpenShift build progress
# Usage: ./scripts/monitor-build.sh [build-name]

BUILD_NAME="${1:-scratchy-spyre}"

echo "🔍 Monitoring build: ${BUILD_NAME}"
echo ""

# Get the latest build number
LATEST_BUILD=$(oc get builds -l buildconfig="${BUILD_NAME}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)

if [ -z "$LATEST_BUILD" ]; then
    echo "❌ No builds found for ${BUILD_NAME}"
    exit 1
fi

echo "📦 Latest build: ${LATEST_BUILD}"
echo ""

# Follow the build logs
oc logs -f "build/${LATEST_BUILD}"

# Check final status
BUILD_STATUS=$(oc get "build/${LATEST_BUILD}" -o jsonpath='{.status.phase}')

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$BUILD_STATUS" = "Complete" ]; then
    echo "✅ Build completed successfully!"
    IMAGE=$(oc get "build/${LATEST_BUILD}" -o jsonpath='{.spec.output.to.name}')
    echo "   Image: ${IMAGE}"
elif [ "$BUILD_STATUS" = "Failed" ]; then
    echo "❌ Build failed!"
    exit 1
else
    echo "⚠️  Build status: ${BUILD_STATUS}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
