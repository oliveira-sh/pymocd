//! External non-dominated archive over `Solution`.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::prism::dense::Solution;

mod add;
mod leader;
mod prune;

pub struct Archive {
    pub members: Vec<Solution>,
    pub capacity: usize,
}

impl Archive {
    pub fn new(capacity: usize) -> Self {
        Archive {
            members: Vec::with_capacity(capacity + capacity / 4 + 1),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }
}
