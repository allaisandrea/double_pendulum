//! The link to the Arduino running `arduino/arduino.ino`.
//!
//! The protocol is one signed byte per command, and the board says nothing
//! but "ready mdd10" at startup. So anything it sends later means it has
//! reset, and the actions around that moment never happened.

use anyhow::{anyhow, bail, Context, Result};
use serialport::{SerialPort, SerialPortType};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const ARDUINO_VID: u16 = 0x2341;
pub const READY: &str = "ready mdd10";
const BAUD: u32 = 115_200;

/// The Uno's port. Its name changes with the USB socket, so it is found by
/// Arduino's vendor id.
pub fn find_port() -> Result<String> {
    let found: Vec<String> = serialport::available_ports()?
        .into_iter()
        .filter(|p| matches!(&p.port_type, SerialPortType::UsbPort(u) if u.vid == ARDUINO_VID))
        // macOS lists each device twice; /dev/cu.* is the one to open.
        .filter(|p| !p.port_name.starts_with("/dev/tty."))
        .map(|p| p.port_name)
        .collect();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => bail!("no Arduino found; plug it in, or pass --port"),
        many => bail!("several Arduinos found ({}); pass --port", many.join(", ")),
    }
}

/// Opens the port, which resets the Uno, and waits for its ready line.
/// Bytes sent before that line would go to the bootloader.
pub fn open(path: &str) -> Result<Box<dyn SerialPort>> {
    let mut port = serialport::new(path, BAUD)
        .timeout(Duration::from_millis(50))
        .open()
        .with_context(|| format!("opening {path}"))?;
    let started = Instant::now();
    let mut seen = Vec::new();
    let mut buf = [0u8; 64];
    while !String::from_utf8_lossy(&seen).contains(READY) {
        if started.elapsed() > Duration::from_secs(5) {
            bail!(
                "no {READY:?} from {path} after 5 s (got {:?}); is arduino/arduino.ino flashed?",
                String::from_utf8_lossy(&seen)
            );
        }
        match port.read(&mut buf) {
            Ok(n) => seen.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => return Err(e).context("reading the ready line"),
        }
    }
    Ok(port)
}

/// Writes one duty byte.
pub fn send(port: &mut dyn SerialPort, duty: i8) -> Result<()> {
    port.write_all(&duty.to_ne_bytes())
        .map_err(|e| anyhow!("writing to the Arduino: {e}"))
}

/// Watches for the board speaking after its ready line, which only happens
/// when it resets. Sets `reset` and keeps what it heard in `heard`.
pub fn watch(
    mut port: Box<dyn SerialPort>,
    stop: &AtomicBool,
    reset: &AtomicBool,
    heard: &Mutex<Vec<u8>>,
) {
    let mut buf = [0u8; 64];
    while !stop.load(Ordering::Relaxed) {
        match port.read(&mut buf) {
            Ok(n) if n > 0 => {
                heard.lock().unwrap().extend_from_slice(&buf[..n]);
                reset.store(true, Ordering::SeqCst);
            }
            _ => {}
        }
    }
}
