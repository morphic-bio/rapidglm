// Links librapidglm_c. RAPIDGLM_LIB_DIR names the directory holding it (a
// RapidGLM build directory or an installed lib directory); otherwise the
// system library path is searched. The directory is also added to the runpath
// of test and example binaries so they run without LD_LIBRARY_PATH.
fn main() {
    println!("cargo:rerun-if-env-changed=RAPIDGLM_LIB_DIR");
    if let Ok(dir) = std::env::var("RAPIDGLM_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    println!("cargo:rustc-link-lib=dylib=rapidglm_c");
}
