// SPDX-License-Identifier: Apache-2.0
//! Every carrier the models crate ships is typed here: the bake asks this crate for each one's
//! shapes, so a carrier without a typed program could not compile.

use scratchy_forward_compiler_frontend::parse_python::parse_python_file;
use scratchy_models_typed::traced;

#[test]
fn every_carrier_has_a_typed_program() {
    let dsl = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../arch/dsl");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dsl).expect("the carriers") {
        let path = entry.expect("a carrier").path();
        if path.extension().is_none_or(|e| e != "py") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read the carrier");
        let carrier =
            parse_python_file(&source).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            traced(&carrier.arch_name).is_some(),
            "{}: [{}] has no typed program",
            path.display(),
            carrier.arch_name
        );
        checked += 1;
    }
    assert!(checked > 0, "no carrier under {}", dsl.display());
}
