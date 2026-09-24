//! Finding, opening and draining a camera.

use crate::clock;
use crate::mailbox::Latest;
use anyhow::{anyhow, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
};
use nokhwa::{Buffer, Camera};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::time::Duration;

/// A camera chosen with --camera.
pub struct Picked {
    pub index: u32,
    pub name: String,
    /// AVFoundation's unique id. macOS can list cameras in a different order
    /// from one query to the next, so the camera is opened by id, not index.
    pub id: String,
}

/// Resolves --camera, an index or a case-insensitive part of a camera's name,
/// and lists the cameras on offer. Indices can change as cameras are plugged
/// in, so a name is the safer choice.
pub fn pick_camera(spec: &str) -> Result<Picked> {
    let cameras: Vec<Picked> = nokhwa::query(ApiBackend::Auto)?
        .iter()
        .filter_map(|c| {
            Some(Picked {
                index: c.index().as_index().ok()?,
                name: c.human_name(),
                id: c.misc(),
            })
        })
        .collect();
    let list = cameras
        .iter()
        .map(|c| format!("[{}] {}", c.index, c.name))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("cameras: {list}");
    let wanted = spec.to_lowercase();
    let found = match spec.parse::<u32>() {
        Ok(i) => cameras.into_iter().find(|c| c.index == i),
        Err(_) => cameras
            .into_iter()
            .find(|c| c.name.to_lowercase().contains(&wanted)),
    };
    found.ok_or_else(|| anyhow!("no camera matches {spec:?}; available: {list}"))
}

fn open_camera(id: &str) -> Result<Camera> {
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);
    let mut cam = Camera::new(CameraIndex::String(id.to_string()), format)?;
    cam.open_stream()?;
    Ok(cam)
}

/// A raw camera frame, untouched until the detector takes it, so frames
/// dropped for being late cost nothing.
pub struct Captured {
    pub seq: u64,
    /// When the sensor captured the frame, on [`clock::mono`]. `None` if the
    /// backend gave no timestamp.
    pub t_capture: Option<Duration>,
    /// When the frame reached this program, on [`clock::mono`].
    pub t_arrival: Duration,
    pub buf: Buffer,
}

/// Whether the frame is packed YUYV at its nominal size. nokhwa labels some
/// other macOS layouts (NV12) YUYV too; the size check keeps those off the
/// fast paths.
pub fn is_packed_yuyv(buf: &Buffer) -> bool {
    let res = buf.resolution();
    buf.source_frame_format() == FrameFormat::YUYV
        && buf.buffer().len() == 2 * res.width() as usize * res.height() as usize
}

/// Describes a camera format the way the programs print it.
pub fn describe(f: &CameraFormat) -> String {
    format!("{}x{} {:?} @ {} fps", f.width(), f.height(), f.format(), f.frame_rate())
}

/// Drains the camera continuously. nokhwa hands out the oldest queued frame
/// and discards the rest, so reading slowly would mean reading stale frames.
/// Each frame replaces the one waiting in `slot`, so the consumer always gets
/// the newest; frames replaced before anyone took them are counted as dropped.
/// The slot is closed when capture stops.
pub fn capture_loop(
    id: &str,
    stop: &AtomicBool,
    slot: &Latest<Captured>,
    dropped: &AtomicU64,
    ready: Sender<Result<CameraFormat>>,
) -> Result<()> {
    let mut cam = match open_camera(id) {
        Ok(cam) => {
            let _ = ready.send(Ok(cam.camera_format()));
            cam
        }
        Err(e) => {
            slot.close();
            let _ = ready.send(Err(e.context(format!("opening camera {id}"))));
            return Ok(());
        }
    };

    let mut seq = 0;
    let result = (|| -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            let buf = cam.frame()?;
            let t_arrival = clock::mono();
            let t_capture = buf.capture_timestamp();
            let replaced = slot.put(Captured {
                seq,
                t_capture,
                t_arrival,
                buf,
            });
            if replaced {
                dropped.fetch_add(1, Ordering::Relaxed);
            }
            seq += 1;
        }
        Ok(())
    })();
    let _ = cam.stop_stream();
    slot.close();
    result
}
