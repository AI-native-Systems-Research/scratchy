fn main() {
    println!("cargo:rerun-if-env-changed=SENLIB_STATIC_LIB_DIR");
    if let Ok(dir) = std::env::var("SENLIB_STATIC_LIB_DIR") {
        println!("cargo:rustc-link-arg-bins=-L{dir}");
        println!("cargo:rustc-link-arg-bins=-l:libflex_shim.a");
        println!("cargo:rustc-link-arg-bins=-l:libsenlib_oco.a");
    }
}
