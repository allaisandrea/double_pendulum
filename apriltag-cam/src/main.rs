//! Records webcam video with AprilTag detections and their poses drawn on
//! top, plus a CSV of every detection, for judging how well tags track.
//!
//! Three threads, so a slow stage never stalls the camera:
//!   capture -> [1 slot, newest wins] -> detect -> [2 slots] -> render
//! Detection reads luma straight from the camera's YUYV bytes; colour
//! conversion, drawing and JPEG encoding all happen on the render thread,
//! off the path that produces poses. Frames the detector cannot keep up
//! with are dropped at the first hand-off and counted; every detected frame
//! is recorded, stamped with its capture time so playback runs in real time.

mod detect;
mod draw;
mod mov;
mod overlay;
#[cfg(test)]
mod synthetic;
mod yuyv;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use detect::{Intrinsics, Pixels, Tag, Tracker};
use image::RgbImage;
use mov::MovWriter;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use nokhwa::{Buffer, Camera};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Record webcam video with AprilTag detections and poses overlaid.
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Args {
    /// Output video (QuickTime, Motion-JPEG); a CSV of detections is written
    /// alongside it [default: recordings/apriltag-<unix time>.mov]
    #[arg(long)]
    out: Option<PathBuf>,

    /// Stop after this many seconds [default: run until Ctrl-C]
    #[arg(long)]
    duration: Option<f64>,

    /// Edge of the tag's black square, in metres (0.8x the sheet's nominal size)
    #[arg(long, default_value_t = 0.040)]
    tag_size: f64,

    /// Tag family
    #[arg(long, default_value = "tag36h11")]
    family: String,

    /// Horizontal field of view in degrees, for nominal intrinsics when
    /// --fx/--fy are not given
    #[arg(long, default_value_t = 60.0)]
    hfov: f64,

    /// Calibrated focal length along x, in pixels
    #[arg(long, requires = "fy")]
    fx: Option<f64>,

    /// Calibrated focal length along y, in pixels
    #[arg(long, requires = "fx")]
    fy: Option<f64>,

    /// Principal point x, in pixels [default: image centre]
    #[arg(long, requires = "cy")]
    cx: Option<f64>,

    /// Principal point y, in pixels [default: image centre]
    #[arg(long, requires = "cx")]
    cy: Option<f64>,

    /// Camera index
    #[arg(long, default_value_t = 0)]
    camera: u32,

    /// Detector threads
    #[arg(long, default_value_t = 4)]
    threads: u8,

    /// Detector decimation: quads are searched for on an image this many
    /// times smaller (decoding still uses full resolution)
    #[arg(long, default_value_t = 2.0)]
    decimate: f32,

    /// JPEG quality of recorded frames
    #[arg(long, default_value_t = 85, value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: u8,
}

impl Args {
    fn intrinsics(&self, width: u32, height: u32) -> Intrinsics {
        let nominal = Intrinsics::from_hfov(width, height, self.hfov);
        Intrinsics {
            fx: self.fx.unwrap_or(nominal.fx),
            fy: self.fy.unwrap_or(nominal.fy),
            cx: self.cx.unwrap_or(nominal.cx),
            cy: self.cy.unwrap_or(nominal.cy),
        }
    }
}

/// A raw camera frame, untouched until the detector takes it, so frames
/// dropped for being late cost nothing.
struct Captured {
    seq: u64,
    pts: Duration,
    buf: Buffer,
}

/// A frame with its detections, on its way to be drawn and recorded.
struct Detected {
    pts: Duration,
    buf: Buffer,
    k: Intrinsics,
    tags: Vec<Tag>,
}

/// Whether the frame is packed YUYV at its nominal size. nokhwa labels some
/// other macOS layouts (NV12) YUYV too; the size check keeps those off the
/// fast paths.
fn is_packed_yuyv(buf: &Buffer) -> bool {
    let res = buf.resolution();
    buf.source_frame_format() == FrameFormat::YUYV
        && buf.buffer().len() == 2 * res.width() as usize * res.height() as usize
}

#[derive(Default)]
struct Stats {
    frames: u64,
    frames_with_tags: u64,
    per_id: BTreeMap<usize, u64>,
    detect_ms: Vec<f64>,
    last_pts: Duration,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let out = args.out.clone().unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        PathBuf::from(format!("recordings/apriltag-{secs}.mov"))
    });
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let csv = out.with_extension("csv");

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        // The first Ctrl-C finishes the file cleanly; a second one bails out.
        ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
        })?;
    }
    let dropped = Arc::new(AtomicU64::new(0));
    let processed = Arc::new(AtomicU64::new(0));

    let (cap_tx, cap_rx) = mpsc::sync_channel(1);
    let (enc_tx, enc_rx) = mpsc::sync_channel(2);
    let (ready_tx, ready_rx) = mpsc::channel();

    let capture = {
        let (stop, dropped, index) = (stop.clone(), dropped.clone(), args.camera);
        thread::Builder::new()
            .name("capture".into())
            .spawn(move || capture_loop(index, &stop, cap_tx, &dropped, ready_tx))?
    };
    let format = ready_rx
        .recv()
        .map_err(|_| anyhow!("capture thread exited"))??;
    eprintln!("camera {}: {format}", args.camera);

    let process = {
        let (args, stop, processed, csv) =
            (args.clone(), stop.clone(), processed.clone(), csv.clone());
        thread::Builder::new()
            .name("detect".into())
            .spawn(move || detect_loop(&args, cap_rx, enc_tx, &stop, &processed, &csv))?
    };
    let encode = {
        let (args, out) = (args.clone(), out.clone());
        thread::Builder::new()
            .name("render".into())
            .spawn(move || render_loop(&args, enc_rx, &out))?
    };

    eprintln!("recording to {} (Ctrl-C to stop)", out.display());
    let started = Instant::now();
    let mut hinted = false;
    while !stop.load(Ordering::Relaxed) && !process.is_finished() && !encode.is_finished() {
        if args
            .duration
            .is_some_and(|d| started.elapsed().as_secs_f64() >= d)
        {
            break;
        }
        if !hinted
            && processed.load(Ordering::Relaxed) == 0
            && started.elapsed() > Duration::from_secs(5)
        {
            eprintln!(
                "no frames yet; if this persists, allow camera access for your terminal in \
                 System Settings > Privacy & Security > Camera"
            );
            hinted = true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    stop.store(true, Ordering::SeqCst);

    let stats = process
        .join()
        .map_err(|_| anyhow!("detection thread panicked"))?;
    let frames = encode
        .join()
        .map_err(|_| anyhow!("render thread panicked"))?;
    // Capture can be stuck waiting on a camera that never delivers; the
    // file is already finished by now, so don't wait on it for long.
    let deadline = Instant::now() + Duration::from_secs(1);
    while !capture.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    if capture.is_finished() {
        if let Err(e) = capture
            .join()
            .map_err(|_| anyhow!("capture thread panicked"))?
        {
            eprintln!("capture stopped early: {e:#}");
        }
    }

    let frames = frames.context("writing video")?;
    let stats = stats.context("detection")?;
    summarize(&stats, frames, dropped.load(Ordering::Relaxed), &out, &csv);
    Ok(())
}

fn open_camera(index: u32) -> Result<Camera> {
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);
    let mut cam = Camera::new(CameraIndex::Index(index), format)?;
    cam.open_stream()?;
    Ok(cam)
}

/// Drains the camera continuously. nokhwa hands out the oldest queued frame
/// and discards the rest, so reading slowly would mean reading stale frames.
fn capture_loop(
    index: u32,
    stop: &AtomicBool,
    tx: SyncSender<Captured>,
    dropped: &AtomicU64,
    ready: Sender<Result<String>>,
) -> Result<()> {
    let mut cam = match open_camera(index) {
        Ok(cam) => {
            let f = cam.camera_format();
            let _ = ready.send(Ok(format!(
                "{}x{} {:?} @ {} fps",
                f.width(),
                f.height(),
                f.format(),
                f.frame_rate()
            )));
            cam
        }
        Err(e) => {
            let _ = ready.send(Err(e.context(format!("opening camera {index}"))));
            return Ok(());
        }
    };

    let mut t0 = None;
    let mut seq = 0;
    let result = (|| -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            let buf = cam.frame()?;
            // The sensor's timestamp where the backend reports one.
            let ts = buf.capture_timestamp().unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
            });
            let pts = ts.saturating_sub(*t0.get_or_insert(ts));
            match tx.try_send(Captured { seq, pts, buf }) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    dropped.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Disconnected(_)) => break,
            }
            seq += 1;
        }
        Ok(())
    })();
    let _ = cam.stop_stream();
    result
}

fn detect_loop(
    args: &Args,
    rx: Receiver<Captured>,
    tx: SyncSender<Detected>,
    stop: &AtomicBool,
    processed: &AtomicU64,
    csv_path: &Path,
) -> Result<Stats> {
    let mut csv = BufWriter::new(
        File::create(csv_path).with_context(|| format!("creating {}", csv_path.display()))?,
    );
    writeln!(
        csv,
        "frame,t_s,detect_ms,id,hamming,margin,x_m,y_m,z_m,qw,qx,qy,qz,err,alt_err"
    )?;

    let mut tracker: Option<(Tracker, Intrinsics)> = None;
    let mut stats = Stats::default();

    // Polls rather than blocks so Ctrl-C lands even if the camera stalls.
    while !stop.load(Ordering::Relaxed) {
        let cap = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(cap) => cap,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let res = cap.buf.resolution();
        let (w, h) = (res.width(), res.height());
        if tracker.is_none() {
            let k = args.intrinsics(w, h);
            tracker = Some((
                Tracker::new(&args.family, args.threads, args.decimate, args.tag_size, k)?,
                k,
            ));
        }
        let (tracker, k) = tracker.as_mut().expect("created above");

        let started = Instant::now();
        let (w, h) = (w as usize, h as usize);
        let tags = if is_packed_yuyv(&cap.buf) {
            tracker.detect(w, h, Pixels::Yuyv(cap.buf.buffer()))?
        } else {
            let rgb = cap.buf.decode_image::<RgbFormat>()?;
            tracker.detect(w, h, Pixels::Rgb(rgb.as_raw()))?
        };
        let detect_ms = started.elapsed().as_secs_f64() * 1e3;
        write_csv(&mut csv, cap.seq, cap.pts, detect_ms, &tags)?;

        stats.frames += 1;
        stats.frames_with_tags += u64::from(!tags.is_empty());
        for tag in &tags {
            *stats.per_id.entry(tag.id).or_default() += 1;
        }
        stats.detect_ms.push(detect_ms);
        stats.last_pts = cap.pts;

        processed.fetch_add(1, Ordering::Relaxed);
        let detected = Detected {
            pts: cap.pts,
            buf: cap.buf,
            k: *k,
            tags,
        };
        if tx.send(detected).is_err() {
            break; // the render thread failed; main reports why
        }
    }
    csv.flush()?;
    Ok(stats)
}

fn write_csv(
    out: &mut impl Write,
    seq: u64,
    pts: Duration,
    detect_ms: f64,
    tags: &[Tag],
) -> Result<()> {
    let frame = format!("{seq},{:.6},{detect_ms:.2}", pts.as_secs_f64());
    if tags.is_empty() {
        // One row per frame even when nothing is seen, so detection rates
        // can be computed from the CSV alone.
        writeln!(out, "{frame},,,,,,,,,,,,")?;
    }
    for tag in tags {
        write!(out, "{frame},{},{},{:.1}", tag.id, tag.hamming, tag.margin)?;
        match &tag.pose {
            Some(p) => {
                let q = p.quaternion();
                let alt = tag.alt_err.map_or(String::new(), |e| format!("{e:.4e}"));
                writeln!(
                    out,
                    ",{:.5},{:.5},{:.5},{:.6},{:.6},{:.6},{:.6},{:.4e},{alt}",
                    p.t[0], p.t[1], p.t[2], q[0], q[1], q[2], q[3], p.err
                )?;
            }
            None => writeln!(out, ",,,,,,,,,")?,
        }
    }
    Ok(())
}

/// Draws detections over each frame and JPEG-encodes it into the movie.
/// The file is finished even if this fails part-way, so whatever was
/// recorded stays playable.
fn render_loop(args: &Args, rx: Receiver<Detected>, path: &Path) -> Result<u64> {
    let mut writer: Option<MovWriter> = None;
    let mut frames = 0;
    let result = (|| -> Result<()> {
        let mut spare: Option<RgbImage> = None;
        let mut jpeg = Vec::new();
        for d in rx {
            let res = d.buf.resolution();
            let (w, h) = (res.width(), res.height());
            let mut img = if is_packed_yuyv(&d.buf) {
                let mut img = spare
                    .take()
                    .filter(|i| i.dimensions() == (w, h))
                    .unwrap_or_else(|| RgbImage::new(w, h));
                yuyv::to_rgb(d.buf.buffer(), &mut img);
                img
            } else {
                d.buf.decode_image::<RgbFormat>()?
            };
            for tag in &d.tags {
                overlay::draw_tag(&mut img, tag, &d.k, args.tag_size);
            }

            let (w16, h16) = (u16::try_from(w)?, u16::try_from(h)?);
            if writer.is_none() {
                writer = Some(
                    MovWriter::create(path, w16, h16)
                        .with_context(|| format!("creating {}", path.display()))?,
                );
            }
            jpeg.clear();
            jpeg_encoder::Encoder::new(&mut jpeg, args.quality).encode(
                img.as_raw(),
                w16,
                h16,
                jpeg_encoder::ColorType::Rgb,
            )?;
            writer
                .as_mut()
                .expect("created above")
                .write_frame(&jpeg, d.pts)?;
            frames += 1;
            spare = Some(img);
        }
        Ok(())
    })();
    match writer {
        Some(w) => w.finish()?,
        None => bail!("no frames were recorded"),
    }
    result.map(|()| frames)
}

fn summarize(stats: &Stats, frames: u64, dropped: u64, out: &Path, csv: &Path) {
    let secs = stats.last_pts.as_secs_f64();
    let fps = if secs > 0.0 {
        (frames.saturating_sub(1)) as f64 / secs
    } else {
        0.0
    };
    eprintln!(
        "\nrecorded {frames} frames over {secs:.1} s ({fps:.1} fps) to {}",
        out.display()
    );
    if dropped > 0 {
        eprintln!("  {dropped} camera frames dropped: detection fell behind the camera");
    }
    if !stats.detect_ms.is_empty() {
        let mut ms = stats.detect_ms.clone();
        ms.sort_by(f64::total_cmp);
        eprintln!(
            "  detection time: median {:.1} ms, p95 {:.1} ms",
            ms[ms.len() / 2],
            ms[ms.len() * 95 / 100]
        );
    }
    let pct = |n: u64| 100.0 * n as f64 / stats.frames.max(1) as f64;
    eprintln!(
        "  frames with a tag: {}/{} ({:.1}%)",
        stats.frames_with_tags,
        stats.frames,
        pct(stats.frames_with_tags)
    );
    for (id, n) in &stats.per_id {
        eprintln!("    id {id}: {n} frames ({:.1}%)", pct(*n));
    }
    eprintln!("  detections: {}", csv.display());
}
