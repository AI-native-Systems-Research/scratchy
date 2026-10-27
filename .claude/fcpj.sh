#!/bin/bash
exec > /work/fcpj.log 2>&1
echo "FCPJ-START"
cd /work/front-a-sampling || exit 1
export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/work/pr916/head/target
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
find crates/serving/worker/src crates/targets/spyre/src crates/targets/spyre/ktir/ktir-superdsc/src crates/compiler/subtile -name '*.rs' -exec touch {} +
SCRATCHY_SKIP_SENDNN_CXX=1 cargo check -p scratchy-cli --features spyre-hw,model/granite-3.1-2b-instruct 2>&1 | grep -E "^(error|warning)" -A12 | head -150
echo "FCPJ-CHECK-EXIT ${PIPESTATUS[0]}"
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy -p scratchy-cli --features spyre-hw,model/granite-3.1-2b-instruct --no-deps -- -D warnings 2>&1 | grep -E "^(error|warning)" -A12 | head -150
echo "FCPJ-CLIPPY-EXIT ${PIPESTATUS[0]}"
SCRATCHY_SKIP_SENDNN_CXX=1 cargo test -p scratchy-subtile --features spyre --test mask_laws 2>&1 | tail -25
echo "FCPJ-MASKLAWS-EXIT ${PIPESTATUS[0]}"
SCRATCHY_SKIP_SENDNN_CXX=1 cargo test -p scratchy-target-spyre --features spyre-emu --lib forward_tape 2>&1 | tail -25
echo "FCPJ-TAPE-EXIT ${PIPESTATUS[0]}"
SCRATCHY_SKIP_SENDNN_CXX=1 cargo test -p scratchy-serving-worker --features spyre-hw --lib pad_clamp 2>&1 | tail -25
echo "FCPJ-PADCLAMP-EXIT ${PIPESTATUS[0]}"
echo "FCPJ-DONE"
