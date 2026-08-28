//! Generates pymocd.pyi from #[`gen_stub`_*] annotations.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>
//!
//! Run: cargo run --no-default-features --bin `stub_gen`

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = pymocd::stub_info()?;
    stub.generate()?;
    Ok(())
}
