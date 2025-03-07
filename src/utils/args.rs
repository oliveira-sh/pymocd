//! args.rs
//! Parse arguments from cli or from lib to be a AGArgs struct
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

#[derive(Debug, Clone, PartialEq)]
pub struct AGArgs {
    pub num_gens: usize,
    pub pop_size: usize,
    pub mut_rate: f64,
    pub cross_rate: f64,
    pub debug: i8,

    pub parallelism: bool,
}

impl Default for AGArgs {
    fn default() -> Self {
        AGArgs {
            num_gens: 1000,
            pop_size: 100,
            cross_rate: 0.9,
            mut_rate: 0.1,
            debug: 0,
            parallelism: true,
        }
    }
}

impl AGArgs {
    pub fn lib_args(verbose: i8) -> Self {
        AGArgs {
            num_gens: 0x3E8,
            pop_size: 100,
            mut_rate: 0.2,
            cross_rate: 0.8,
            parallelism: true,
            debug: verbose,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_struct(){
        let default_struct = AGArgs::default();
        let expected_struct =   AGArgs {
            num_gens: 1000,
            pop_size: 100,
            cross_rate: 0.9,
            mut_rate: 0.1,
            debug: 0,
            parallelism: true,
        };
        assert_eq!(default_struct, expected_struct);
    }
}