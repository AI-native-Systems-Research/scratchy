fn main() {
    println!("cargo:rerun-if-env-changed=SENLIB_STATIC_LIB_DIR");
    if let Ok(dir) = std::env::var("SENLIB_STATIC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=static:-bundle=flex_shim");
        println!("cargo:rustc-link-lib=static:-bundle=senlib_oco");
    }
}
