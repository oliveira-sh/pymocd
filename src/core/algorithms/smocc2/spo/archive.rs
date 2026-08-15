use std::collections::HashSet;

use crate::core::algorithms::smocc::nsga2::{
    Obj, crowding_distance, dominates, fast_nondominated_sort,
};

use super::particles::{MacElite, MicElite};

fn archive_select(objs: &[Obj], same: impl Fn(usize, usize) -> bool, cap: usize) -> Vec<usize> {
    let ranks = fast_nondominated_sort(objs);
    let mut keep: Vec<usize> = (0..objs.len()).filter(|&i| ranks[i] == 1).collect();

    let mut by_obj: Vec<usize> = keep.clone();
    by_obj.sort_by(|&a, &b| {
        objs[a]
            .partial_cmp(&objs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut dup: HashSet<usize> = HashSet::new();
    let mut run_start = 0;
    for i in 1..=by_obj.len() {
        if i == by_obj.len() || objs[by_obj[i]] != objs[by_obj[run_start]] {
            for a in run_start..i {
                if dup.contains(&by_obj[a]) {
                    continue;
                }
                for b in (a + 1)..i {
                    if !dup.contains(&by_obj[b]) && same(by_obj[a], by_obj[b]) {
                        dup.insert(by_obj[b]);
                    }
                }
            }
            run_start = i;
        }
    }
    keep.retain(|i| !dup.contains(i));

    if keep.len() > cap {
        let kobjs: Vec<Obj> = keep.iter().map(|&i| objs[i].clone()).collect();
        let crowd = crowding_distance(&kobjs, &vec![1usize; kobjs.len()]);
        let mut order: Vec<usize> = (0..keep.len()).collect();
        order.sort_by(|&a, &b| {
            crowd[b]
                .partial_cmp(&crowd[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(cap);
        order.sort_unstable();
        keep = order.into_iter().map(|i| keep[i]).collect();
    }

    debug_assert!(keep.iter().all(|&a| {
        keep.iter()
            .all(|&b| a == b || !dominates(&objs[b], &objs[a]))
    }));
    keep
}

pub(crate) fn update_micro_archive(
    mut arch: Vec<MicElite>,
    fresh: Vec<MicElite>,
    cap: usize,
) -> Vec<MicElite> {
    arch.extend(fresh);
    let objs: Vec<Obj> = arch.iter().map(|a| a.obj.clone()).collect();
    let keep = archive_select(&objs, |a, b| arch[a].labels == arch[b].labels, cap);
    let keep: HashSet<usize> = keep.into_iter().collect();
    arch.into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, a)| a)
        .collect()
}

pub(crate) fn update_macro_archive(
    mut arch: Vec<MacElite>,
    fresh: Vec<MacElite>,
    cap: usize,
) -> Vec<MacElite> {
    arch.extend(fresh);
    let objs: Vec<Obj> = arch.iter().map(|a| a.obj.clone()).collect();
    let keep = archive_select(&objs, |a, b| arch[a].genome == arch[b].genome, cap);
    let keep: HashSet<usize> = keep.into_iter().collect();
    arch.into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, a)| a)
        .collect()
}

pub(crate) fn arch_crowd(objs: &[Obj]) -> Vec<f64> {
    crowding_distance(objs, &vec![1usize; objs.len()])
}
