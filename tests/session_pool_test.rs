use jetort::engine::session_pool::Pool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn test_pool_basic_checkout_and_return() {
    let pool = Pool::new("test-model", vec![10, 20, 30]);
    assert_eq!(pool.capacity(), 3);
    assert_eq!(pool.available_count(), 3);
    assert_eq!(pool.model_name(), "test-model");

    {
        let mut item1 = pool.acquire();
        assert_eq!(pool.available_count(), 2);
        assert!(*item1 == 30 || *item1 == 20 || *item1 == 10);
        *item1 += 5;
    }
    // After dropping item1, pool has 3 available again
    assert_eq!(pool.available_count(), 3);

    // Verify item was returned
    let item = pool.acquire();
    assert!(pool.available_count() == 2);
    drop(item);
    assert_eq!(pool.available_count(), 3);
}

#[test]
fn test_pool_timeout_when_exhausted() {
    let pool = Pool::new("exhaust-test", vec!["session-1"]);
    let guard = pool.acquire();
    assert_eq!(pool.available_count(), 0);

    // Second checkout with short timeout should fail
    let timeout_res = pool.acquire_timeout(Duration::from_millis(50));
    assert!(timeout_res.is_err());
    let err_msg = timeout_res.err().unwrap().to_string();
    assert!(err_msg.contains("timed out"));

    drop(guard);
    assert_eq!(pool.available_count(), 1);

    // After return, should succeed immediately
    let second_res = pool.acquire_timeout(Duration::from_millis(50));
    assert!(second_res.is_ok());
}

#[test]
fn test_pool_concurrent_multi_thread_stress() {
    let pool = Arc::new(Pool::new("stress-test", vec![1, 2, 3, 4]));
    let counter = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    for _ in 0..20 {
        let p = pool.clone();
        let c = counter.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..50 {
                let guard = p.acquire();
                std::thread::sleep(Duration::from_micros(200));
                c.fetch_add(*guard, Ordering::Relaxed);
                drop(guard);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(pool.available_count(), 4);
    assert!(counter.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_owned_guard_transfer_across_threads() {
    let pool = Pool::new("owned-test", vec!["worker-1"]);
    let owned_guard = pool.acquire_owned();
    assert_eq!(pool.available_count(), 0);

    let handle = std::thread::spawn(move || {
        assert_eq!(*owned_guard, "worker-1");
        // Drop inside new thread
        drop(owned_guard);
    });

    handle.join().unwrap();
    // Verify session returned to pool
    assert_eq!(pool.available_count(), 1);
}
