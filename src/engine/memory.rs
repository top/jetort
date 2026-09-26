use serde::Serialize;
#[allow(unused_imports)]
use std::fs;
use tracing::info;

#[derive(Debug, Clone, Serialize, Default)]
pub struct MemorySnapshot {
    pub process_rss_mb: f64,
    pub process_swap_mb: f64,
    pub process_peak_rss_mb: f64,
    pub system_total_mb: f64,
    pub system_available_mb: f64,
    pub system_swap_total_mb: f64,
    pub system_swap_used_mb: f64,
}

/// Retrieve current system and process memory metrics.
/// On Linux (Jetson Orin/Xavier/Tegra), inspects /proc/self/status and /proc/meminfo.
#[allow(unused_mut)]
pub fn get_memory_snapshot() -> MemorySnapshot {
    let mut snap = MemorySnapshot::default();

    #[cfg(target_os = "linux")]
    {
        // 1. Process memory metrics from /proc/self/status
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        snap.process_rss_mb = kb / 1024.0;
                    }
                } else if let Some(rest) = line.strip_prefix("VmSwap:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        snap.process_swap_mb = kb / 1024.0;
                    }
                } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        snap.process_peak_rss_mb = kb / 1024.0;
                    }
                }
            }
        }

        // 2. System-wide memory metrics from /proc/meminfo
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            let mut swap_total_kb = 0.0;
            let mut swap_free_kb = 0.0;

            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        snap.system_total_mb = kb / 1024.0;
                    }
                } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        snap.system_available_mb = kb / 1024.0;
                    }
                } else if let Some(rest) = line.strip_prefix("SwapTotal:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        swap_total_kb = kb;
                        snap.system_swap_total_mb = kb / 1024.0;
                    }
                } else if let Some(rest) = line.strip_prefix("SwapFree:") {
                    if let Some(kb) = parse_kb_val(rest) {
                        swap_free_kb = kb;
                    }
                }
            }

            if swap_total_kb >= swap_free_kb {
                snap.system_swap_used_mb = (swap_total_kb - swap_free_kb) / 1024.0;
            }
        }
    }

    snap
}

#[cfg(target_os = "linux")]
fn parse_kb_val(s: &str) -> Option<f64> {
    let num_str = s.trim().split_whitespace().next()?;
    num_str.parse::<f64>().ok()
}

/// Trims glibc free heap arenas back to the OS.
pub fn trim_allocator_heap() {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> std::os::raw::c_int;
        }
        unsafe {
            let _ = malloc_trim(0);
        }
    }
}

/// Log a structured memory telemetry milestone, optionally comparing against a previous snapshot.
pub fn log_memory_milestone(label: &str, before: Option<&MemorySnapshot>) -> MemorySnapshot {
    let now = get_memory_snapshot();

    if let Some(prev) = before {
        let delta_rss = now.process_rss_mb - prev.process_rss_mb;

        info!(
            milestone = %label,
            rss_mb = now.process_rss_mb,
            delta_rss_mb = delta_rss,
            swap_mb = now.process_swap_mb,
            system_available_mb = now.system_available_mb,
            system_total_mb = now.system_total_mb,
            system_swap_used_mb = now.system_swap_used_mb,
            "memory milestone"
        );
    } else {
        info!(
            milestone = %label,
            rss_mb = now.process_rss_mb,
            swap_mb = now.process_swap_mb,
            system_available_mb = now.system_available_mb,
            system_total_mb = now.system_total_mb,
            system_swap_used_mb = now.system_swap_used_mb,
            "memory milestone"
        );
    }

    now
}
