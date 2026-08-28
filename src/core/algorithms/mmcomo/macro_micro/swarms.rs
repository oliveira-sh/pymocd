//! The micro and macro population members and the NSGA-II bookkeeping over them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mmcomo::nsga2::{
    crowding_distance, environment_selection, fast_nondominated_sort,
};
use crate::core::algorithms::mmcomo::{Genome, Labels};

#[derive(Clone)]
pub struct Mic {
    pub labels: Labels,
    pub obj: (f64, f64),
}

#[derive(Clone)]
pub struct Mac {
    pub genome: Genome,
    pub labels: Labels,
    pub obj: (f64, f64),
}

pub fn micro_objs(p: &[Mic]) -> Vec<(f64, f64)> {
    p.iter().map(|x| x.obj).collect()
}

pub fn macro_objs(p: &[Mac]) -> Vec<(f64, f64)> {
    p.iter().map(|x| x.obj).collect()
}

pub fn ranks_and_crowd(objs: &[(f64, f64)]) -> (Vec<usize>, Vec<f64>) {
    let ranks = fast_nondominated_sort(objs);
    let crowd = crowding_distance(objs, &ranks);
    (ranks, crowd)
}

pub fn select_micro(pool: Vec<Mic>, keep: usize) -> Vec<Mic> {
    let objs = micro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

pub fn select_macro(pool: Vec<Mac>, keep: usize) -> Vec<Mac> {
    let objs = macro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}
