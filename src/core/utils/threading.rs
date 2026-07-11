//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::ThreadPoolBuilder;
use std::sync::Once;

static INIT_RAYON: Once = Once::new();

#[gen_stub_pyfunction]
#[pyfunction]
pub fn max_cores(num_threads: usize) -> PyResult<()> {
    INIT_RAYON.call_once(|| {
        ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
        debug!(warn, "Global thread pool initialized initialized with {} threads", num_threads);
        debug!(warn, "Using max_cores again has no effect, due to static ThreadPoolBuilder initialization")
    });
    Ok(())
}
