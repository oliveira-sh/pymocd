//! Operator-mode configuration resolving `topo_mode` into the micro variation
//! operators each generation enables.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const TOPO_MAJORITY_MUT: u8 = 1 << 1;

pub const TOPO_HPMOCD_CROSS: u8 = 1 << 7;

const MICRO_BITS: u8 = TOPO_MAJORITY_MUT | TOPO_HPMOCD_CROSS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MicroOps {
    pub majority_mut: bool,
    pub hpmocd_cross: bool,
}

impl MicroOps {
    pub fn from_topo(topo_mode: u8) -> Self {
        let t = topo_mode & MICRO_BITS;
        MicroOps {
            majority_mut: t & TOPO_MAJORITY_MUT != 0,
            hpmocd_cross: t & TOPO_HPMOCD_CROSS != 0,
        }
    }

    pub fn any(self) -> bool {
        self != MicroOps::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_ops_decodes_only_the_two_live_bits() {
        assert_eq!(TOPO_MAJORITY_MUT, 1 << 1);
        assert_eq!(TOPO_HPMOCD_CROSS, 1 << 7);
        assert_eq!(MICRO_BITS, 0b1000_0010);

        assert_eq!(MicroOps::from_topo(0), MicroOps::default());
        assert!(!MicroOps::from_topo(0).any());
        assert!(MicroOps::from_topo(TOPO_MAJORITY_MUT).majority_mut);
        assert!(MicroOps::from_topo(TOPO_HPMOCD_CROSS).hpmocd_cross);

        let shipped = MicroOps::from_topo(130);
        assert!(shipped.majority_mut && shipped.hpmocd_cross);

        for dead in [1u8, 4, 8, 16, 32, 64] {
            assert!(
                !MicroOps::from_topo(dead).any(),
                "deleted bit {dead} still routes micro"
            );
            assert_eq!(
                MicroOps::from_topo(130 | dead),
                shipped,
                "deleted bit {dead} changed the shipped mask"
            );
        }
        assert_eq!(MicroOps::from_topo(0xFF), shipped);
    }

    #[test]
    fn micro_routing_only_reacts_to_micro_bits() {
        for mask in [0u8, 1, 4, 8, 16, 32, 64] {
            assert!(!MicroOps::from_topo(mask).any(), "mask {mask} routes micro");
            assert_eq!(mask & MICRO_BITS, 0);
        }
        for mask in [2u8, 128, 130] {
            assert!(MicroOps::from_topo(mask).any(), "mask {mask} skips micro");
            assert_eq!(mask & MICRO_BITS, mask);
        }
    }
}
