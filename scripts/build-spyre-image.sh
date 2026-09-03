#!/bin/bash
set -euo pipefail

# Configuration
IMAGE_TAG="${1:-latest}"
BUILD_NAME="scratchy-spyre"
NAMESPACE=$(oc project -q)
IMAGE_REGISTRY="${IMAGE_REGISTRY:-image-registry.openshift-image-registry.svc:5000/${NAMESPACE}}"
FULL_IMAGE="${IMAGE_REGISTRY}/scratchy-spyre:${IMAGE_TAG}"

echo "🚀 Building Scratchy Spyre image in OpenShift cluster"
echo "   Build name: ${BUILD_NAME}"
echo "   Target image: ${FULL_IMAGE}"
echo ""

# Check if we're logged into OpenShift
if ! oc whoami &>/dev/null; then
    echo "❌ Not logged into OpenShift. Please run 'oc login' first."
    exit 1
fi

# Check if build already exists
if oc get bc "${BUILD_NAME}" &>/dev/null; then
    echo "📦 Build config '${BUILD_NAME}' already exists (will reuse for layer caching)"
else
    echo "📝 Creating BuildConfig from tools/docker/Dockerfile.spyre..."
    cat <<EOF | oc apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: ${BUILD_NAME}
  labels:
    app: scratchy
    component: spyre
spec:
  output:
    to:
      kind: DockerImage
      name: ${FULL_IMAGE}
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: Dockerfile.spyre
EOF
    echo ""
    echo "✅ BuildConfig created"
    echo ""
fi

# Create minimal build context
echo "📦 Creating minimal build context..."
BUILD_CONTEXT=$(mktemp -d)
trap "rm -rf ${BUILD_CONTEXT}" EXIT

cp tools/docker/Dockerfile.spyre "${BUILD_CONTEXT}/"
cp Cargo.toml Cargo.lock build.rs "${BUILD_CONTEXT}/"
cp -r crates "${BUILD_CONTEXT}/"
# Copy third_party but exclude test directories
mkdir -p "${BUILD_CONTEXT}/third_party"
rsync -a --exclude='tests/' --exclude='test/' --exclude='build/' --exclude='*.o' --exclude='*.a' third_party/ "${BUILD_CONTEXT}/third_party/"

# Start build
echo "🔨 Starting build (uploading ~50MB instead of full repo)..."
oc start-build "${BUILD_NAME}" --from-dir="${BUILD_CONTEXT}" --follow

echo ""
echo "✅ Build complete!"
echo "   Image: ${FULL_IMAGE}"
echo ""
echo "To use this image:"
echo "  oc new-app ${FULL_IMAGE}"
echo "  or"
echo "  podman pull ${FULL_IMAGE}"
