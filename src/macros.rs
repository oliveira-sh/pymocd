//! xfeats.rs
//! Implements macros for the library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
#[macro_export]
macro_rules! debug {
    // invocation: debug!(debug  , "something {} happened", x);
    //            debug!(warn, "watch out: {}", y);
    //            debug!(err  , "failed: {:?}", err);
    ($level:ident, $($arg:tt)*) => {
        {
            let (lvl, color) = match stringify!($level) {
                "debug"   => ("DEBUG"  , "\x1b[34m"),
                "warn" => ("WARNING", "\x1b[33m"),
                "err"   => ("ERROR"  , "\x1b[31m"),
                other     => (other   , "\x1b[0m"),
            };

            let file = file!();
            let line = line!();
            println!(
                "{}[ {}: {}:{} {}]\x1b[0m {}",
                color, lvl, file, line, color,
                format_args!($($arg)*)
            );
        }
    };
}
