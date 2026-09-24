//! Records RL training data: tag poses from the camera, and the actions a
//! policy sends to the motor, on one monotonic clock.
//!
//!   capture -> [newest frame wins] -> detect -> [every detection] -> policy -> serial
//!                                                                      \-> logger
//!
//! Capture drains the camera and keeps only the newest frame, so detection
//! never works on a stale one. Detection passes every result on. The policy
//! thread takes everything that has arrived, acts on the newest frame, and
//! writes the action to the motor at once. Frames it skipped while busy go
//! into its history and the table all the same, with the action that was in
//! effect. Nothing waits on a clock: an action goes out as soon as it is
//! ready, and its send time is recorded.
//!
//! For now the policy is a stand-in, a random walk; see `policy.rs`.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use harness::camera::{self, capture_loop, pick_camera, Frame};
use harness::clock::mono;
use harness::constants::{HFOV_DEG, TAG_FAMILY, TAG_SIZE_M};
use harness::latest::{Latest, Take};
use harness::policy::{DutyCycle, Policy, RandomWalkPolicy, Step, HISTORY_LENGTH};
use harness::table::{self, FrameRow, Table, TagRow};
use harness::tag_detector::{Intrinsics, TagDetector};
use harness::{serial, uvc};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Record tag poses and motor actions for RL, on one clock.
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Args {
    /// Output directory [default: recordings/<unix time>]
    #[arg(long)]
    out: Option<PathBuf>,

    /// Stop after this many seconds [default: run until Ctrl-C]
    #[arg(long)]
    duration: Option<f64>,

    /// Stand-in policy: the action walks within -range..=range. 0 sends only
    /// zeros, so nothing moves
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u8).range(0..=127))]
    policy_range: u8,

    /// Stand-in policy: each frame the action moves by a step drawn
    /// uniformly from -step..=step. It takes about 3 * (range / step)^2
    /// frames to wander from 0 to a limit
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u8).range(0..=127))]
    policy_step: u8,

    /// Stand-in policy: simulated inference time per action, in
    /// milliseconds; the camera delivers a frame every 8.3 ms
    #[arg(long, default_value_t = 7.0)]
    policy_latency_ms: f64,

    /// The stand-in policy is on for this long, then rests (all zeros) for
    /// --rest-s, and repeats, so one recording holds both driven motion and
    /// the arm settling afterwards. Seconds of capture time from t0
    #[arg(long, default_value_t = 20.0)]
    active_s: f64,

    /// Rest between active periods, in seconds; 0 never rests. At zero duty
    /// the shield brakes the motor, so the arm settles damped, not free
    #[arg(long, default_value_t = 5.0)]
    rest_s: f64,

    /// Seed for the stand-in policy [default: random, printed and recorded]
    #[arg(long)]
    seed: Option<u64>,

    /// Print a status line this often, in seconds; 0 prints only the final
    /// summary
    #[arg(long, default_value_t = 10.0)]
    status_every_s: f64,

    /// Serial port of the Arduino [default: found by USB vendor id]
    #[arg(long)]
    port: Option<String>,

    /// Camera: an index, or part of its name
    #[arg(long, default_value = "Arducam")]
    camera: String,

    /// Fixed exposure in microseconds; 0 leaves it to the camera. The
    /// default was measured best on this rig; see the README
    #[arg(long, default_value_t = 200)]
    exposure_us: u32,

    /// Sensor gain, 0..100
    #[arg(long, default_value_t = 75)]
    gain: u16,

    /// Detector threads. Two keep up with 120 fps on the M4 (about 5 ms a
    /// frame) and leave the other cores to the policy
    #[arg(long, default_value_t = 2)]
    threads: u8,

    /// Detector decimation
    #[arg(long, default_value_t = 2.0)]
    decimate: f32,
}

/// A frame through detection, on its way to the policy thread.
struct Detected {
    frame: u64,
    t_capture: Duration,
    t_arrival: Duration,
    t_detect_start: Duration,
    t_detected: Duration,
    /// Tags 0, 1 and 2, by id.
    tags: [Option<TagRow>; 3],
}

fn main() -> Result<()> {
    let args = Args::parse();
    let seed = args.seed.unwrap_or_else(rand::random);
    anyhow::ensure!(
        args.active_s > 0.0 && args.rest_s >= 0.0 && args.policy_latency_ms >= 0.0,
        "--active-s must be positive; --rest-s and --policy-latency-ms not negative"
    );
    let duty = DutyCycle {
        active: Duration::from_secs_f64(args.active_s),
        rest: Duration::from_secs_f64(args.rest_s),
    };
    let policy = RandomWalkPolicy {
        seed,
        range: args.policy_range as i8,
        step: args.policy_step,
        latency: Duration::from_secs_f64(args.policy_latency_ms / 1e3),
        duty,
    };

    let out = args.out.clone().unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        PathBuf::from(format!("recordings/{secs}"))
    });
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        // The first Ctrl-C stops cleanly, motor first; a second one bails out
        // and leaves the board's watchdog to stop the motor.
        ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
        })?;
    }

    // The board first: opening its port resets it, and it takes ~1.5 s to
    // come back, which overlaps with the camera starting up below.
    let port_path = match &args.port {
        Some(p) => p.clone(),
        None => serial::find_port()?,
    };
    let mut port = serial::open(&port_path)?;
    serial::send(port.as_mut(), 0)?;
    eprintln!("arduino: {port_path} ({})", serial::READY);

    let cam = pick_camera(&args.camera)?;
    let exposure = if args.exposure_us > 0 {
        let dev = uvc::Device::find(&cam.id).context("finding the camera's UVC device")?;
        let applied = dev
            .set_exposure(args.exposure_us, Some(args.gain))
            .context("setting a fixed exposure")?;
        eprintln!(
            "exposure: {} us fixed, gain {}",
            applied.exposure_us, args.gain
        );
        Some((dev, applied))
    } else {
        None
    };

    // Every timestamp is recorded relative to t0, before the first frame.
    let t0 = mono();
    let wall_t0 = SystemTime::now().duration_since(UNIX_EPOCH)?;

    let dropped = Arc::new(AtomicU64::new(0));
    let slot = Arc::new(Latest::new());
    let opened = camera::open(&cam.id)?;
    let format = opened.camera_format();
    let capture = {
        let (stop, dropped, slot) = (stop.clone(), dropped.clone(), slot.clone());
        thread::Builder::new()
            .name("capture".into())
            .spawn(move || capture_loop(opened, &stop, &slot, &dropped))?
    };
    let k = Intrinsics::from_hfov(format.width(), format.height(), HFOV_DEG);
    eprintln!("camera: {} ({})", cam.name, camera::describe(&format));

    let meta: HashMap<String, String> = [
        ("t0_mono_ns", t0.as_nanos().to_string()),
        ("t0_unix_ns", wall_t0.as_nanos().to_string()),
        ("clock", "CLOCK_UPTIME_RAW (mach_absolute_time), ns".into()),
        (
            "pose",
            "x y z qw qx qy qz: tag frame to camera frame, metres; qw >= 0".into(),
        ),
        (
            "tag_frame",
            "origin at tag centre, x right, y down, z into the tag".into(),
        ),
        ("tags", "[0, 1, 2]".into()),
        ("tag_family", TAG_FAMILY.into()),
        ("tag_size_m", TAG_SIZE_M.to_string()),
        (
            "intrinsics",
            format!("fx={} fy={} cx={} cy={}", k.fx, k.fy, k.cx, k.cy),
        ),
        (
            "camera",
            format!("{} ({})", cam.name, camera::describe(&format)),
        ),
        ("exposure_us", args.exposure_us.to_string()),
        ("gain", args.gain.to_string()),
        ("decimate", args.decimate.to_string()),
        ("history", HISTORY_LENGTH.to_string()),
        ("policy", policy.describe()),
        ("policy_range", args.policy_range.to_string()),
        ("policy_step", args.policy_step.to_string()),
        ("policy_latency_ns", policy.latency.as_nanos().to_string()),
        ("active_ns", duty.active.as_nanos().to_string()),
        ("rest_ns", duty.rest.as_nanos().to_string()),
        ("seed", seed.to_string()),
        ("board", serial::READY.into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    let frames = Table::create(&out.join("frames.arrows"), meta)?;
    let (row_tx, row_rx) = mpsc::channel::<FrameRow>();
    let status_every = Duration::from_secs_f64(args.status_every_s.max(0.0));
    let logger = thread::Builder::new()
        .name("logger".into())
        .spawn(move || log_loop(row_rx, frames, duty, status_every))?;

    let reset = Arc::new(AtomicBool::new(false));
    let heard = Arc::new(Mutex::new(Vec::new()));
    let watcher = {
        let port = port.try_clone().context("cloning the serial port")?;
        let (stop, reset, heard) = (stop.clone(), reset.clone(), heard.clone());
        thread::Builder::new()
            .name("serial-watch".into())
            .spawn(move || serial::watch(port, &stop, &reset, &heard))?
    };

    let (det_tx, det_rx) = mpsc::channel::<Detected>();
    let detect = {
        let (args, stop) = (args.clone(), stop.clone());
        thread::Builder::new()
            .name("detect".into())
            .spawn(move || detect_loop(&args, k, &slot, det_tx, &stop))?
    };

    let act = {
        let stop = stop.clone();
        thread::Builder::new()
            .name("policy".into())
            .spawn(move || policy_loop(Box::new(policy), det_rx, port, t0, row_tx, &stop))?
    };

    let cycle = if duty.rest.is_zero() {
        "always on".to_string()
    } else {
        format!("{} s on, {} s rest", args.active_s, args.rest_s)
    };
    eprintln!(
        "recording to {} (seed {seed}, policy ±{} step {}, {cycle}); Ctrl-C to stop",
        out.display(),
        args.policy_range,
        args.policy_step
    );
    let started = Instant::now();
    let mut abort = None;
    while !stop.load(Ordering::Relaxed) {
        if args
            .duration
            .is_some_and(|d| started.elapsed().as_secs_f64() >= d)
        {
            break;
        }
        if reset.load(Ordering::Relaxed) {
            abort = Some(format!(
                "the Arduino reset mid-run (it sent {:?}); actions around that moment \
                 were not applied, so this recording is unreliable",
                String::from_utf8_lossy(&heard.lock().unwrap())
            ));
            break;
        }
        if act.is_finished() || detect.is_finished() || logger.is_finished() {
            break; // a thread failed; the joins below say why
        }
        thread::sleep(Duration::from_millis(50));
    }
    stop.store(true, Ordering::SeqCst);

    // The policy thread first: it writes a final zero, so the motor stops
    // before anything else winds down.
    let acted = act.join().map_err(|_| anyhow!("policy thread panicked"))?;
    let detected = detect
        .join()
        .map_err(|_| anyhow!("detect thread panicked"))?;
    let stats = logger.join().map_err(|_| anyhow!("logger panicked"))?;
    let _ = watcher.join();
    let deadline = Instant::now() + Duration::from_secs(1);
    while !capture.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }

    acted.context("policy")?;
    detected.context("detection")?;
    let stats = stats.context("writing the recording")?;
    stats.report(dropped.load(Ordering::Relaxed), &out);
    if let Some((dev, applied)) = &exposure {
        if let Some(what) = dev.drift(applied) {
            eprintln!("warning: exposure did not hold during this recording: {what}");
        }
    }
    if let Some(why) = abort {
        bail!(why);
    }
    Ok(())
}

/// Detects tags in the newest frame, over and over, and passes every result
/// to the policy thread.
fn detect_loop(
    args: &Args,
    k: Intrinsics,
    slot: &Latest<Frame>,
    to_policy: Sender<Detected>,
    stop: &AtomicBool,
) -> Result<()> {
    let mut detector = TagDetector::new(TAG_FAMILY, args.threads, args.decimate, TAG_SIZE_M, k)?;
    while !stop.load(Ordering::Relaxed) {
        let frame = match slot.take(Duration::from_millis(100)) {
            Take::Item(frame) => frame,
            Take::Timeout => continue,
            Take::Closed => break,
        };
        let t_capture = frame
            .t_capture
            .ok_or_else(|| anyhow!("frame {} has no capture timestamp", frame.seq))?;

        let res = frame.buf.resolution();
        let (w, h) = (res.width() as usize, res.height() as usize);
        let t_detect_start = mono();
        let tags = detector.detect(w, h, frame.buf.buffer())?;
        let t_detected = mono();

        let tag_rows: [Option<TagRow>; 3] = std::array::from_fn(|id| {
            let tag = tags.iter().find(|t| t.id == id)?;
            let pose = tag.pose.as_ref()?;
            Some(TagRow {
                pose: table::pose_row(pose.t, pose.quaternion()),
                err: pose.err as f32,
                alt_err: tag.alt_err.map(|e| e as f32),
                margin: tag.margin,
            })
        });

        let sent = to_policy.send(Detected {
            frame: frame.seq,
            t_capture,
            t_arrival: frame.t_arrival,
            t_detect_start,
            t_detected,
            tags: tag_rows,
        });
        if sent.is_err() {
            break; // the policy thread has stopped
        }
    }
    Ok(())
}

/// Takes every detection that has arrived, acts on the newest, and sends
/// the action at once. The older ones were skipped while the policy was
/// busy: they join the history, and the table, with the action in effect.
fn policy_loop(
    mut policy: Box<dyn Policy>,
    from_detect: Receiver<Detected>,
    mut port: Box<dyn serialport::SerialPort>,
    t0: Duration,
    rows: Sender<FrameRow>,
    stop: &AtomicBool,
) -> Result<()> {
    raise_priority();
    let ns = |t: Duration| table::rel_ns(t, t0);
    let step = |d: &Detected, action: Option<i8>| Step {
        frame: d.frame,
        t: d.t_capture.saturating_sub(t0),
        poses: d.tags.map(|t| t.map(|t| t.pose)),
        action,
    };
    let row = |d: Detected, action: i8, times: Option<[Duration; 3]>| FrameRow {
        frame: d.frame,
        t_capture: ns(d.t_capture),
        t_arrival: ns(d.t_arrival),
        t_detect_start: ns(d.t_detect_start),
        t_detected: ns(d.t_detected),
        tags: d.tags,
        acted: times.is_some(),
        t_policy_start: times.map(|t| ns(t[0])),
        t_policy_done: times.map(|t| ns(t[1])),
        t_sent: times.map(|t| ns(t[2])),
        action,
    };
    // The frames before the newest, oldest first, each with the action in
    // effect after it.
    let mut history: VecDeque<Step> = VecDeque::with_capacity(HISTORY_LENGTH);
    let remember = |history: &mut VecDeque<Step>, s: Step| {
        if history.len() == HISTORY_LENGTH - 1 {
            history.pop_front();
        }
        history.push_back(s);
    };
    let mut current: i8 = 0;

    let result = (|| -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            let first = match from_detect.recv_timeout(Duration::from_millis(100)) {
                Ok(d) => d,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            let mut waiting = vec![first];
            waiting.extend(from_detect.try_iter());
            let newest = waiting.pop().expect("at least the first");
            for d in waiting {
                remember(&mut history, step(&d, Some(current)));
                let _ = rows.send(row(d, current, None));
            }

            if history.len() < HISTORY_LENGTH - 1 {
                // The start of the recording: this frame only fills the
                // history, and the startup 0 stays in effect.
                remember(&mut history, step(&newest, Some(current)));
                let _ = rows.send(row(newest, current, None));
                continue;
            }

            let t_policy_start = mono();
            // Newest first: this frame, then the history from its end.
            let input: [Step; HISTORY_LENGTH] = std::array::from_fn(|i| match i {
                0 => step(&newest, None),
                _ => history[history.len() - i].clone(),
            });
            let action = policy.act(&input);
            let t_policy_done = mono();
            serial::send(port.as_mut(), action)?;
            let t_sent = mono();
            current = action;
            remember(&mut history, step(&newest, Some(action)));
            let _ = rows.send(row(
                newest,
                action,
                Some([t_policy_start, t_policy_done, t_sent]),
            ));
        }
        Ok(())
    })();
    // Stop the motor whatever happened above.
    let zeroed = serial::send(port.as_mut(), 0).and_then(|_| Ok(port.flush()?));
    result.and(zeroed)
}

/// Runs the calling thread at the highest ordinary priority, so the policy
/// keeps up when other work competes for the cores.
fn raise_priority() {
    // SAFETY: affects only the calling thread; no pointers involved.
    let rc = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
    if rc != 0 {
        eprintln!("warning: could not raise the policy thread's priority (error {rc})");
    }
}

/// Writes the frames table, a batch about once a second, until the policy
/// thread hangs up. Also keeps the run's statistics, from the same rows it
/// writes: a status line every `status_every` (zero prints none), and the
/// totals, returned at the end.
fn log_loop(
    rx: Receiver<FrameRow>,
    mut frames: Table,
    duty: DutyCycle,
    status_every: Duration,
) -> Result<Stats> {
    let started = Instant::now();
    let (mut total, mut interval) = (Stats::default(), Stats::default());
    let mut next_flush = started + Duration::from_secs(1);
    let mut next_status = started + status_every;
    loop {
        let wake = match status_every.is_zero() {
            true => next_flush,
            false => next_flush.min(next_status),
        };
        match rx.recv_timeout(wake.saturating_duration_since(Instant::now())) {
            Ok(r) => {
                // The first rows only fill the policy's history; after them,
                // a row not acted on was skipped while the policy was busy.
                let startup = total.frames < HISTORY_LENGTH as u64 - 1;
                total.add(&r, startup);
                interval.add(&r, startup);
                frames.push(r);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        let now = Instant::now();
        if now >= next_flush {
            frames.flush()?;
            next_flush += Duration::from_secs(1);
        }
        if !status_every.is_zero() && now >= next_status {
            eprintln!("{}", interval.status_line(started.elapsed(), duty));
            interval = Stats::default();
            next_status += status_every;
        }
    }
    frames.finish()?;
    Ok(total)
}

/// An account of frame rows: of a whole run, or of one status interval.
#[derive(Default)]
struct Stats {
    frames: u64,
    /// Frames tags 0, 1 and 2 were each seen in.
    with_tag: [u64; 3],
    acted: u64,
    /// Frames not acted on because the policy was busy, not counting the
    /// startup frames that only fill its history.
    skipped: u64,
    /// Capture time of the latest row, since t0, in nanoseconds.
    last_capture: Option<i64>,
    /// Milliseconds: capture to arrival, detection, queued between detection
    /// and policy, inference, and capture to the action written.
    age_ms: Vec<f64>,
    detect_ms: Vec<f64>,
    wait_ms: Vec<f64>,
    policy_ms: Vec<f64>,
    delay_ms: Vec<f64>,
}

impl Stats {
    fn add(&mut self, r: &FrameRow, startup: bool) {
        let ms = |a: i64, b: i64| (b - a) as f64 / 1e6;
        self.frames += 1;
        for (n, tag) in self.with_tag.iter_mut().zip(&r.tags) {
            *n += u64::from(tag.is_some());
        }
        self.last_capture = Some(r.t_capture);
        self.age_ms.push(ms(r.t_capture, r.t_arrival));
        self.detect_ms.push(ms(r.t_detect_start, r.t_detected));
        match (r.t_policy_start, r.t_policy_done, r.t_sent) {
            (Some(start), Some(done), Some(sent)) => {
                self.acted += 1;
                self.wait_ms.push(ms(r.t_detected, start));
                self.policy_ms.push(ms(start, done));
                self.delay_ms.push(ms(r.t_capture, sent));
            }
            _ if !startup => self.skipped += 1,
            _ => {}
        }
    }

    /// One line on this interval, `elapsed` into the run.
    fn status_line(&self, elapsed: Duration, duty: DutyCycle) -> String {
        let phase = match self.last_capture {
            _ if duty.rest.is_zero() => "",
            Some(t) if duty.resting(Duration::from_nanos(t.max(0) as u64)) => " resting |",
            Some(_) => " active  |",
            None => "",
        };
        let tags: Vec<String> = self
            .with_tag
            .iter()
            .map(|n| format!("{:.0}", 100.0 * *n as f64 / self.frames.max(1) as f64))
            .collect();
        let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
        format!(
            "[{:5.0} s]{phase} {} frames, tags {}% | {} actions, {} frames skipped | \
             slowest detection {:.0} ms, slowest capture to send {:.0} ms",
            elapsed.as_secs_f64(),
            self.frames,
            tags.join("/"),
            self.acted,
            self.skipped,
            max(&self.detect_ms),
            max(&self.delay_ms),
        )
    }

    /// The summary printed at the end of a run.
    fn report(&self, dropped: u64, out: &std::path::Path) {
        eprintln!("\nwrote {} frames to {}", self.frames, out.display());
        eprintln!("frames: {dropped} camera frames dropped before detection");
        for (id, n) in self.with_tag.iter().enumerate() {
            eprintln!(
                "  tag {id}: in {n} frames ({:.1}%)",
                100.0 * *n as f64 / self.frames.max(1) as f64
            );
        }
        eprintln!(
            "  frame age on arrival (clock check): {}",
            quantiles(&mut self.age_ms.clone())
        );
        eprintln!("  detection: {}", quantiles(&mut self.detect_ms.clone()));
        eprintln!(
            "policy: acted on {} frames after the first {}, skipped {} ({:.1}%) while busy",
            self.acted,
            HISTORY_LENGTH - 1,
            self.skipped,
            100.0 * self.skipped as f64 / (self.acted + self.skipped).max(1) as f64
        );
        eprintln!(
            "  queued after detection: {}",
            quantiles(&mut self.wait_ms.clone())
        );
        eprintln!("  inference: {}", quantiles(&mut self.policy_ms.clone()));
        eprintln!(
            "  capture to send: {}",
            quantiles(&mut self.delay_ms.clone())
        );
    }
}

fn quantiles(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "none".into();
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!(
        "min {:.1}  median {:.1}  p95 {:.1}  max {:.1} ms",
        q(0.0),
        q(0.5),
        q(0.95),
        q(1.0)
    )
}
