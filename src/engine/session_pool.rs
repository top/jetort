use crate::error::AppError;
use parking_lot::{Condvar, Mutex};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Duration;

struct PoolInner<T> {
    items: Mutex<Vec<T>>,
    condvar: Condvar,
    capacity: usize,
    name: String,
}

/// A thread-safe, RAII-managed concurrency pool for resources (such as ONNX Runtime Sessions).
/// Allows concurrent inference across multiple worker threads while enforcing
/// memory safety and configurable concurrency limits on edge hardware (e.g. Jetson).
pub struct Pool<T> {
    inner: Arc<PoolInner<T>>,
}

impl<T> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

pub struct Guard<'a, T> {
    item: Option<T>,
    pool: &'a Pool<T>,
}

impl<'a, T> Deref for Guard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Guard item missing")
    }
}

impl<'a, T> DerefMut for Guard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().expect("Guard item missing")
    }
}

impl<'a, T> Drop for Guard<'a, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let mut lock = self.pool.inner.items.lock();
            lock.push(item);
            self.pool.inner.condvar.notify_one();
        }
    }
}

/// Owned variant of Guard that can be moved across asynchronous boundaries or blocking threads.
pub struct OwnedGuard<T> {
    item: Option<T>,
    pool: Pool<T>,
}

impl<T> Deref for OwnedGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("OwnedGuard item missing")
    }
}

impl<T> DerefMut for OwnedGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().expect("OwnedGuard item missing")
    }
}

impl<T> Drop for OwnedGuard<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let mut lock = self.pool.inner.items.lock();
            lock.push(item);
            self.pool.inner.condvar.notify_one();
        }
    }
}

impl<T> Pool<T> {
    /// Creates a new Pool from a pre-initialized vector of items.
    pub fn new(name: impl Into<String>, items: Vec<T>) -> Self {
        let capacity = items.len();
        Self {
            inner: Arc::new(PoolInner {
                items: Mutex::new(items),
                condvar: Condvar::new(),
                capacity,
                name: name.into(),
            }),
        }
    }

    /// Alias for creating from sessions
    pub fn from_sessions(name: impl Into<String>, sessions: Vec<T>) -> Self {
        Self::new(name, sessions)
    }

    /// Acquires an item from the pool, blocking the calling thread until one becomes available.
    pub fn acquire(&self) -> Guard<'_, T> {
        let mut lock = self.inner.items.lock();
        while lock.is_empty() {
            self.inner.condvar.wait(&mut lock);
        }
        let item = lock.pop().expect("Item present after condvar wakeup");
        Guard {
            item: Some(item),
            pool: self,
        }
    }

    /// Acquires an owned guard, allowing transfer across blocking thread tasks.
    pub fn acquire_owned(&self) -> OwnedGuard<T> {
        let mut lock = self.inner.items.lock();
        while lock.is_empty() {
            self.inner.condvar.wait(&mut lock);
        }
        let item = lock.pop().expect("Item present after condvar wakeup");
        OwnedGuard {
            item: Some(item),
            pool: self.clone(),
        }
    }

    /// Acquires an item with a timeout.
    pub fn acquire_timeout(&self, timeout: Duration) -> Result<Guard<'_, T>, AppError> {
        let mut lock = self.inner.items.lock();
        let start = std::time::Instant::now();
        while lock.is_empty() {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(AppError::Inference(format!(
                    "Resource checkout timed out after {:?} for '{}'",
                    timeout, self.inner.name
                )));
            }
            let remaining = timeout - elapsed;
            let res = self.inner.condvar.wait_for(&mut lock, remaining);
            if res.timed_out() && lock.is_empty() {
                return Err(AppError::Inference(format!(
                    "Resource checkout timed out after {:?} for '{}'",
                    timeout, self.inner.name
                )));
            }
        }
        let item = lock.pop().expect("Item present after wait");
        Ok(Guard {
            item: Some(item),
            pool: self,
        })
    }

    /// Acquires an owned item with a timeout.
    pub fn acquire_owned_timeout(&self, timeout: Duration) -> Result<OwnedGuard<T>, AppError> {
        let guard = self.acquire_timeout(timeout)?;
        let mut g = guard;
        let item = g.item.take().expect("Item must be present in guard");
        Ok(OwnedGuard {
            item: Some(item),
            pool: self.clone(),
        })
    }

    /// Number of currently idle items in the pool.
    pub fn available_count(&self) -> usize {
        self.inner.items.lock().len()
    }

    /// Total capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Name associated with this pool.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Model name associated with this pool (alias for name).
    pub fn model_name(&self) -> &str {
        &self.inner.name
    }
}

pub type SessionPool = Pool<ort::session::Session>;
pub type SessionGuard<'a> = Guard<'a, ort::session::Session>;
pub type OwnedSessionGuard = OwnedGuard<ort::session::Session>;
