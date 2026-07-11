//! Ground-truth agreement metrics between two node→community dicts.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::core::graph::to_partition;
use crate::core::metrics::{ami, ari, contingency, entropy, f1, mutual_info, nmi};

/// Align two {node:community} dicts into label vectors over their shared nodes.
fn aligned_labels(
    partition: &Bound<'_, PyDict>,
    gt: &Bound<'_, PyDict>,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let p = to_partition(partition)?;
    let g = to_partition(gt)?;
    let mut a = Vec::with_capacity(p.len());
    let mut b = Vec::with_capacity(p.len());
    for (node, &c) in &p {
        if let Some(&cg) = g.get(node) {
            a.push(c as i64);
            b.push(cg as i64);
        }
    }
    Ok((a, b))
}

/// (NMI, AMI, ARI, F1) of a partition against ground truth (gt).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "gt_metrics", signature = (partition, gt))]
pub fn gt_metrics_fn(
    partition: &Bound<'_, PyDict>,
    gt: &Bound<'_, PyDict>,
) -> PyResult<(f64, f64, f64, f64)> {
    let (a, b) = aligned_labels(partition, gt)?;
    Ok(crate::core::metrics::gt_metrics(&a, &b))
}

/// Normalised mutual information between two {node:community} dicts.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "nmi", signature = (partition, gt))]
pub fn nmi_fn(partition: &Bound<'_, PyDict>, gt: &Bound<'_, PyDict>) -> PyResult<f64> {
    let (a, b) = aligned_labels(partition, gt)?;
    let ct = contingency(&a, &b);
    let mi = mutual_info(&ct);
    Ok(nmi::nmi(
        mi,
        entropy(&ct.rows, ct.n),
        entropy(&ct.cols, ct.n),
    ))
}

/// Adjusted mutual information between two {node:community} dicts.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "ami", signature = (partition, gt))]
pub fn ami_fn(partition: &Bound<'_, PyDict>, gt: &Bound<'_, PyDict>) -> PyResult<f64> {
    let (a, b) = aligned_labels(partition, gt)?;
    let ct = contingency(&a, &b);
    let mi = mutual_info(&ct);
    Ok(ami::ami(
        &ct,
        mi,
        entropy(&ct.rows, ct.n),
        entropy(&ct.cols, ct.n),
    ))
}

/// Adjusted Rand index between two {node:community} dicts.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "ari", signature = (partition, gt))]
pub fn ari_fn(partition: &Bound<'_, PyDict>, gt: &Bound<'_, PyDict>) -> PyResult<f64> {
    let (a, b) = aligned_labels(partition, gt)?;
    Ok(ari::ari(&contingency(&a, &b)))
}

/// Pairwise F1 between two {node:community} dicts.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "f1", signature = (partition, gt))]
pub fn f1_fn(partition: &Bound<'_, PyDict>, gt: &Bound<'_, PyDict>) -> PyResult<f64> {
    let (a, b) = aligned_labels(partition, gt)?;
    Ok(f1::f1(&contingency(&a, &b)))
}
