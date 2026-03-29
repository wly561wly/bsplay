fn main() {
    #[cfg(target_os = "linux")]
    {
        // ort_sys / onnxruntime may require C++ runtime symbols on Linux.
        println!("cargo:rustc-link-lib=dylib=stdc++");

        // Provide missing __isoc23_strto* symbols on older glibc.
        cc::Build::new()
            .file("src/isoc23_compat.c")
            .compile("isoc23_compat");
    }

    #[cfg(feature = "gui")]
    tauri_build::build();
}
