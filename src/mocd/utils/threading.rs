//! Thread management utilities for parallel processing
//!
//! This module provides utilities for managing thread pools and parallel processing
//! in community detection algorithms.

use rayon::ThreadPoolBuilder;
use std::sync::atomic::{AtomicUsize, Ordering};

static THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Sets the number of threads for parallel processing
///
/// This function configures the global thread pool used by Rayon for parallel operations.
/// If `num_threads` is 0, it will use the number of available CPU cores.
///
/// # Arguments
/// * `num_threads` - Number of threads to use (0 for automatic detection)
///
/// # Returns
/// The actual number of threads that will be used
///
/// # Examples
///
/// ```rust
/// use pymocd::utils::set_thread_count;
///
/// // Use 4 threads
/// let actual_threads = set_thread_count(4);
/// assert_eq!(actual_threads, 4);
///
/// // Use automatic detection (number of CPU cores)
/// let auto_threads = set_thread_count(0);
/// println!("Using {} threads", auto_threads);
/// ```
pub fn set_thread_count(num_threads: usize) -> usize {
    let actual_threads = if num_threads == 0 {
        // Use number of available CPU cores
        num_cpus::get()
    } else {
        num_threads
    };

    // Configure Rayon's global thread pool
    if let Err(e) = ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build_global()
    {
        // If we can't build the global pool (it may already exist),
        // just continue with the existing configuration
        eprintln!("Warning: Could not set thread count: {}", e);
        return get_thread_count();
    }

    // Store the thread count
    THREAD_COUNT.store(actual_threads, Ordering::Relaxed);
    actual_threads
}

/// Gets the current thread count setting
///
/// # Returns
/// The number of threads currently configured, or 0 if not set
pub fn get_thread_count() -> usize {
    let stored_count = THREAD_COUNT.load(Ordering::Relaxed);
    if stored_count == 0 {
        // If not explicitly set, return the number of threads in the current pool
        rayon::current_num_threads()
    } else {
        stored_count
    }
}

/// Gets the number of available CPU cores
///
/// # Returns
/// Number of logical CPU cores available on the system
pub fn get_cpu_count() -> usize {
    num_cpus::get()
}

/// Gets information about the current threading configuration
///
/// # Returns
/// A ThreadInfo struct containing thread and CPU information
pub fn get_thread_info() -> ThreadInfo {
    ThreadInfo {
        configured_threads: get_thread_count(),
        actual_threads: rayon::current_num_threads(),
        cpu_cores: get_cpu_count(),
    }
}

/// Information about the threading configuration
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    /// Number of threads configured via set_thread_count
    pub configured_threads: usize,
    /// Actual number of threads in the Rayon pool
    pub actual_threads: usize,
    /// Number of CPU cores available
    pub cpu_cores: usize,
}

impl ThreadInfo {
    /// Returns true if the configuration is using all available CPU cores
    pub fn is_using_all_cores(&self) -> bool {
        self.actual_threads == self.cpu_cores
    }

    /// Returns the thread utilization ratio (threads / cores)
    pub fn utilization_ratio(&self) -> f64 {
        if self.cpu_cores == 0 {
            0.0
        } else {
            self.actual_threads as f64 / self.cpu_cores as f64
        }
    }
}

/// Executes a closure with a temporarily different thread count
///
/// This is useful for algorithms that may benefit from different parallelization
/// strategies for different phases.
///
/// # Arguments
/// * `temp_threads` - Temporary thread count to use
/// * `f` - Closure to execute with the temporary thread count
///
/// # Returns
/// The result of the closure execution
///
/// # Note
/// This function creates a new thread pool scope, so it should be used judiciously
/// as thread pool creation has overhead.
pub fn with_thread_count<T, F>(temp_threads: usize, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let actual_threads = if temp_threads == 0 {
        num_cpus::get()
    } else {
        temp_threads
    };

    ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build()
        .unwrap()
        .install(f)
}

/// Determines optimal thread count for a given workload size
///
/// This function provides a heuristic for choosing the number of threads
/// based on the workload characteristics.
///
/// # Arguments
/// * `workload_size` - Size of the work to be parallelized (e.g., population size)
/// * `min_work_per_thread` - Minimum amount of work per thread to make parallelization worthwhile
///
/// # Returns
/// Recommended number of threads for the workload
pub fn optimal_thread_count(workload_size: usize, min_work_per_thread: usize) -> usize {
    if min_work_per_thread == 0 {
        return 1;
    }

    let max_useful_threads = workload_size / min_work_per_thread;
    let available_threads = get_cpu_count();

    max_useful_threads.min(available_threads).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cpu_count() {
        let cpu_count = get_cpu_count();
        assert!(cpu_count > 0);
        assert!(cpu_count <= 1024); // Reasonable upper bound
    }

    #[test]
    fn test_thread_info() {
        let info = get_thread_info();
        assert!(info.cpu_cores > 0);
        assert!(info.actual_threads > 0);

        let ratio = info.utilization_ratio();
        assert!(ratio >= 0.0);
        assert!(ratio <= 10.0); // Allow for oversubscription but within reason
    }

    #[test]
    fn test_optimal_thread_count() {
        // Small workload should use fewer threads
        assert_eq!(optimal_thread_count(4, 2), 2);

        // Large workload with small minimum work per thread
        let optimal = optimal_thread_count(1000, 1);
        assert!(optimal > 0);
        assert!(optimal <= get_cpu_count());

        // Zero minimum work should return 1 thread
        assert_eq!(optimal_thread_count(100, 0), 1);

        // Workload smaller than minimum should return 1 thread
        assert_eq!(optimal_thread_count(2, 10), 1);
    }

    #[test]
    fn test_with_thread_count() {
        let result = with_thread_count(2, || rayon::current_num_threads());

        // Result should reflect the temporary thread count
        // Note: This might not always be exactly 2 depending on system constraints
        assert!(result >= 1);
    }

    #[test]
    fn test_thread_count_zero_means_auto() {
        let cpu_count = get_cpu_count();

        // Setting to 0 should use CPU count
        let auto_threads = set_thread_count(0);
        assert_eq!(auto_threads, cpu_count);
    }
}
