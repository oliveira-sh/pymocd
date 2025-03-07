//! operators/metrics.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub struct Metrics {
    pub modularity: f64,
    pub intra: f64,
    pub inter: f64,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            modularity: 0.0,
            intra: 0.0,
            inter: 0.0,
        }
    }
}

impl Metrics {
    #[allow(dead_code)]
    pub fn get_modularity(&self) -> f64 {
        self.modularity
    }
}
