//! Generates pymocd.pyi from #[gen_stub_*] annotations.
//! Run: cargo run --no-default-features --bin stub_gen

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = pymocd::stub_info()?;
    stub.generate()?;
    Ok(())
}
