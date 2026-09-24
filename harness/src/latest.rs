//! A one-slot mailbox where the newest item wins.
//!
//! The producer overwrites whatever is waiting, so the consumer always takes
//! the most recent item, however long it has been away. That is what a camera
//! hand-off needs: a channel with room for one frame keeps the frame that was
//! already waiting and refuses newer ones, so a consumer that idles between
//! takes gets a frame as old as its idle time.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

pub struct Latest<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
}

/// The outcome of [`Latest::take`].
pub enum Take<T> {
    Item(T),
    Timeout,
    /// The producer has closed the mailbox and nothing is waiting.
    Closed,
}

impl<T> Default for Latest<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Latest<T> {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                value: None,
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    /// Puts `item` in the slot. Returns true if it replaced one nobody took.
    pub(crate) fn put(&self, item: T) -> bool {
        let replaced = self.state.lock().unwrap().value.replace(item);
        self.ready.notify_one();
        // Dropped here, outside the lock: a frame is megabytes.
        replaced.is_some()
    }

    /// Takes the waiting item, waiting up to `timeout` for one to arrive.
    pub fn take(&self, timeout: Duration) -> Take<T> {
        let state = self.state.lock().unwrap();
        let (mut state, _) = self
            .ready
            .wait_timeout_while(state, timeout, |s| s.value.is_none() && !s.closed)
            .unwrap();
        match state.value.take() {
            Some(item) => Take::Item(item),
            None if state.closed => Take::Closed,
            None => Take::Timeout,
        }
    }

    /// Tells the consumer no more items will come. One still waiting is
    /// delivered first.
    pub(crate) fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.ready.notify_all();
    }
}

struct State<T> {
    value: Option<T>,
    closed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn item<T>(t: Take<T>) -> Option<T> {
        match t {
            Take::Item(v) => Some(v),
            _ => None,
        }
    }

    #[test]
    fn the_newest_item_wins() {
        let m = Latest::new();
        assert!(!m.put(1));
        assert!(m.put(2)); // replaced 1, which nobody took
        assert!(m.put(3));
        assert_eq!(item(m.take(Duration::ZERO)), Some(3));
        assert!(matches!(m.take(Duration::ZERO), Take::Timeout));
    }

    #[test]
    fn take_waits_for_an_item() {
        let m = Arc::new(Latest::new());
        let producer = {
            let m = m.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                m.put(7);
            })
        };
        assert_eq!(item(m.take(Duration::from_secs(5))), Some(7));
        producer.join().unwrap();
    }

    #[test]
    fn closing_delivers_what_is_waiting_then_reports_closed() {
        let m = Latest::new();
        m.put(1);
        m.close();
        assert_eq!(item(m.take(Duration::ZERO)), Some(1));
        assert!(matches!(m.take(Duration::from_secs(5)), Take::Closed));
    }
}
