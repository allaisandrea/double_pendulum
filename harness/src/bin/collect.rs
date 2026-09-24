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
use harness::camera::{self, capture_loop, is_packed_yuyv, pick_camera, Captured};
use harness::clock::mono;
use harness::detect::{Intrinsics, Pixels, Tracker};
use harness::mailbox::{Latest, Take};
use harness::policy::{DutyCycle, Policy, RandomWalk, Step, HISTORY};
use harness::record::{self, FrameRow, Table, TagRow};
use harness::{serial, uvc};
use clap::Parser;
use nokhwa::pixel_format::RgbFormat;
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

    /// Tag ids to record, one column group each
    #[arg(long, value_delimiter = ',', default_value = "0,1,2")]
    tags: Vec<usize>,

    /// Edge of the tag's black square, in metres
    #[arg(long, default_value_t = 0.023)]
    tag_size: f64,

    /// Tag family
    #[arg(long, default_value = "tag36h11")]
    family: String,

    /// Horizontal field of view in degrees, when --fx/--fy are not given.
    /// Estimated from the rig's measured camera distance; see the README
    #[arg(long, default_value_t = 72.0)]
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

    /// Detector threads. Two keep up with 120 fps on the M4 (about 5 ms a
    /// frame) and leave the other cores to the policy
    #[arg(long, default_value_t = 2)]
    threads: u8,

    /// Detector decimation
    #[arg(long, default_value_t = 2.0)]
    decimate: f32,
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

/// A frame through detection, on its way to the policy thread.
struct Detected {
    frame: u64,
    t_capture: Duration,
    t_arrival: Duration,
    t_detect_start: Duration,
    t_detected: Duration,
    tags: Vec<Option<TagRow>>,
}

/// Counters the threads bump as they go, read by the periodic status line.
struct Live {
    frames: AtomicU64,
    /// Frames each recorded tag was seen in, in `--tags` order.
    tags: Vec<AtomicU64>,
    acted: AtomicU64,
    skipped: AtomicU64,
    /// Slowest detection and slowest capture-to-send since the last status
    /// line, in microseconds; the status line resets them.
    slowest_detect_us: AtomicU64,
    slowest_send_us: AtomicU64,
}

impl Live {
    fn new(n_tags: usize) -> Self {
        Self {
            frames: AtomicU64::new(0),
            tags: (0..n_tags).map(|_| AtomicU64::new(0)).collect(),
            acted: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            slowest_detect_us: AtomicU64::new(0),
            slowest_send_us: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> Counts {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        Counts {
            frames: get(&self.frames),
            tags: self.tags.iter().map(get).collect(),
            acted: get(&self.acted),
            skipped: get(&self.skipped),
        }
    }
}

#[derive(Default)]
struct Counts {
    frames: u64,
    tags: Vec<u64>,
    acted: u64,
    skipped: u64,
}

/// One line covering the interval since `prev`.
fn status_line(live: &Live, prev: &Counts, now: &Counts, elapsed: Duration, phase: &str) -> String {
    let frames = now.frames - prev.frames;
    let tags: Vec<String> = now
        .tags
        .iter()
        .zip(&prev.tags)
        .map(|(n, p)| format!("{:.0}", 100.0 * (n - p) as f64 / frames.max(1) as f64))
        .collect();
    let ms = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / 1e3;
    format!(
        "[{:5.0} s]{phase} {frames} frames, tags {}% | {} actions, {} frames skipped | \
         slowest detection {:.0} ms, slowest capture to send {:.0} ms",
        elapsed.as_secs_f64(),
        tags.join("/"),
        now.acted - prev.acted,
        now.skipped - prev.skipped,
        ms(&live.slowest_detect_us),
        ms(&live.slowest_send_us),
    )
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
    let policy = RandomWalk {
        seed,
        range: args.policy_range as i8,
        step: args.policy_step,
        latency: Duration::from_secs_f64(args.policy_latency_ms / 1e3),
        duty,
    };

    let out = args.out.clone().unwrap_or_else(|| {
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
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
        eprintln!("exposure: {} us fixed, gain {}", applied.exposure_us, args.gain);
        Some((dev, applied))
    } else {
        None
    };

    // Every timestamp is recorded relative to t0, before the first frame.
    let t0 = mono();
    let wall_t0 = SystemTime::now().duration_since(UNIX_EPOCH)?;

    let dropped = Arc::new(AtomicU64::new(0));
    let slot = Arc::new(Latest::new());
    let (ready_tx, ready_rx) = mpsc::channel();
    let capture = {
        let (stop, dropped, id, slot) =
            (stop.clone(), dropped.clone(), cam.id.clone(), slot.clone());
        thread::Builder::new()
            .name("capture".into())
            .spawn(move || capture_loop(&id, &stop, &slot, &dropped, ready_tx))?
    };
    let format = ready_rx.recv().map_err(|_| anyhow!("capture thread exited"))??;
    let k = args.intrinsics(format.width(), format.height());
    eprintln!("camera: {} ({})", cam.name, camera::describe(&format));

    let meta: HashMap<String, String> = [
        ("t0_mono_ns", t0.as_nanos().to_string()),
        ("t0_unix_ns", wall_t0.as_nanos().to_string()),
        ("clock", "CLOCK_UPTIME_RAW (mach_absolute_time), ns".into()),
        ("pose", "x y z qw qx qy qz: tag frame to camera frame, metres; qw >= 0".into()),
        ("tag_frame", "origin at tag centre, x right, y down, z into the tag".into()),
        ("tags", format!("{:?}", args.tags)),
        ("tag_family", args.family.clone()),
        ("tag_size_m", args.tag_size.to_string()),
        ("intrinsics", format!("fx={} fy={} cx={} cy={}", k.fx, k.fy, k.cx, k.cy)),
        ("camera", format!("{} ({})", cam.name, camera::describe(&format))),
        ("exposure_us", args.exposure_us.to_string()),
        ("gain", args.gain.to_string()),
        ("decimate", args.decimate.to_string()),
        ("history", HISTORY.to_string()),
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

    let frames = Table::create(&out.join("frames.arrows"), &args.tags, meta)?;
    let (row_tx, row_rx) = mpsc::channel::<FrameRow>();
    let logger = thread::Builder::new()
        .name("logger".into())
        .spawn(move || log_loop(row_rx, frames))?;

    let live = Arc::new(Live::new(args.tags.len()));

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
        let (args, stop, live) = (args.clone(), stop.clone(), live.clone());
        thread::Builder::new()
            .name("detect".into())
            .spawn(move || detect_loop(&args, k, &slot, det_tx, &live, &stop))?
    };

    let act = {
        let (stop, live) = (stop.clone(), live.clone());
        thread::Builder::new()
            .name("policy".into())
            .spawn(move || policy_loop(Box::new(policy), det_rx, port, t0, row_tx, &live, &stop))?
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
    let status_every = Duration::from_secs_f64(args.status_every_s.max(0.0));
    let mut next_status = started + status_every;
    let mut prev = Counts {
        tags: vec![0; args.tags.len()],
        ..Counts::default()
    };
    while !stop.load(Ordering::Relaxed) {
        if !status_every.is_zero() && Instant::now() >= next_status {
            let now = live.snapshot();
            let phase = match duty.rest.is_zero() {
                true => "",
                false if duty.resting(mono() - t0) => " resting |",
                false => " active  |",
            };
            eprintln!("{}", status_line(&live, &prev, &now, started.elapsed(), phase));
            prev = now;
            next_status += status_every;
        }
        if args.duration.is_some_and(|d| started.elapsed().as_secs_f64() >= d) {
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
    let detected = detect.join().map_err(|_| anyhow!("detect thread panicked"))?;
    let logged = logger.join().map_err(|_| anyhow!("logger panicked"))?;
    let _ = watcher.join();
    let deadline = Instant::now() + Duration::from_secs(1);
    while !capture.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }

    let acted = acted.context("policy")?;
    let detected = detected.context("detection")?;
    let n_rows = logged.context("writing the recording")?;
    report(&detected, &acted, dropped.load(Ordering::Relaxed), n_rows, &out);
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

#[derive(Default)]
struct DetectStats {
    frames: u64,
    with_tag: HashMap<usize, u64>,
    detect_ms: Vec<f64>,
    /// How old each frame was when it arrived: arrival minus capture.
    age_ms: Vec<f64>,
}

/// Detects tags in the newest frame, over and over, and passes every result
/// to the policy thread.
fn detect_loop(
    args: &Args,
    k: Intrinsics,
    slot: &Latest<Captured>,
    to_policy: Sender<Detected>,
    live: &Live,
    stop: &AtomicBool,
) -> Result<DetectStats> {
    let mut tracker = Tracker::new(&args.family, args.threads, args.decimate, args.tag_size, k)?;
    let mut stats = DetectStats::default();

    while !stop.load(Ordering::Relaxed) {
        let cap = match slot.take(Duration::from_millis(100)) {
            Take::Item(cap) => cap,
            Take::Timeout => continue,
            Take::Closed => break,
        };
        let t_capture = cap
            .t_capture
            .ok_or_else(|| anyhow!("frame {} has no capture timestamp", cap.seq))?;
        stats.age_ms.push((cap.t_arrival.as_secs_f64() - t_capture.as_secs_f64()) * 1e3);

        let res = cap.buf.resolution();
        let (w, h) = (res.width() as usize, res.height() as usize);
        let t_detect_start = mono();
        let tags = if is_packed_yuyv(&cap.buf) {
            tracker.detect(w, h, Pixels::Yuyv(cap.buf.buffer()))?
        } else {
            let rgb = cap.buf.decode_image::<RgbFormat>()?;
            tracker.detect(w, h, Pixels::Rgb(rgb.as_raw()))?
        };
        let t_detected = mono();
        stats.frames += 1;
        stats.detect_ms.push((t_detected - t_detect_start).as_secs_f64() * 1e3);
        live.frames.fetch_add(1, Ordering::Relaxed);
        live.slowest_detect_us
            .fetch_max((t_detected - t_detect_start).as_micros() as u64, Ordering::Relaxed);

        let tag_rows: Vec<Option<TagRow>> = args
            .tags
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let tag = tags.iter().find(|t| t.id == id)?;
                let pose = tag.pose.as_ref()?;
                *stats.with_tag.entry(id).or_default() += 1;
                live.tags[i].fetch_add(1, Ordering::Relaxed);
                Some(TagRow {
                    pose: record::pose_row(pose.t, pose.quaternion()),
                    err: pose.err as f32,
                    alt_err: tag.alt_err.map(|e| e as f32),
                    margin: tag.margin,
                })
            })
            .collect();

        let sent = to_policy.send(Detected {
            frame: cap.seq,
            t_capture,
            t_arrival: cap.t_arrival,
            t_detect_start,
            t_detected,
            tags: tag_rows,
        });
        if sent.is_err() {
            break; // the policy thread has stopped
        }
    }
    Ok(stats)
}

#[derive(Default)]
struct PolicyStats {
    acted: u64,
    skipped: u64,
    /// Detection done to policy start: time spent queued.
    wait_ms: Vec<f64>,
    /// Policy start to action ready.
    policy_ms: Vec<f64>,
    /// Capture to the action written: the command delay.
    delay_ms: Vec<f64>,
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
    live: &Live,
    stop: &AtomicBool,
) -> Result<PolicyStats> {
    raise_priority();
    let mut stats = PolicyStats::default();
    let ns = |t: Duration| record::rel_ns(t, t0);
    let ms = |a: Duration, b: Duration| (b.as_secs_f64() - a.as_secs_f64()) * 1e3;
    let step = |d: &Detected, action: Option<i8>| Step {
        frame: d.frame,
        t: d.t_capture.saturating_sub(t0),
        poses: d.tags.iter().map(|t| t.as_ref().map(|t| t.pose)).collect(),
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
    // Earlier frames, oldest first, each with the action in effect after it.
    let mut history: VecDeque<Step> = VecDeque::with_capacity(HISTORY);
    let remember = |history: &mut VecDeque<Step>, s: Step| {
        if history.len() == HISTORY - 1 {
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
                stats.skipped += 1;
                live.skipped.fetch_add(1, Ordering::Relaxed);
                remember(&mut history, step(&d, Some(current)));
                let _ = rows.send(row(d, current, None));
            }

            let t_policy_start = mono();
            let input: Vec<Step> = std::iter::once(step(&newest, None))
                .chain(history.iter().rev().cloned())
                .collect();
            let action = policy.act(&input);
            let t_policy_done = mono();
            serial::send(port.as_mut(), action)?;
            let t_sent = mono();
            current = action;

            stats.acted += 1;
            stats.wait_ms.push(ms(newest.t_detected, t_policy_start));
            stats.policy_ms.push(ms(t_policy_start, t_policy_done));
            stats.delay_ms.push(ms(newest.t_capture, t_sent));
            live.acted.fetch_add(1, Ordering::Relaxed);
            live.slowest_send_us
                .fetch_max(t_sent.saturating_sub(newest.t_capture).as_micros() as u64, Ordering::Relaxed);
            remember(&mut history, step(&newest, Some(action)));
            let _ = rows.send(row(newest, action, Some([t_policy_start, t_policy_done, t_sent])));
        }
        Ok(())
    })();
    // Stop the motor whatever happened above.
    let zeroed = serial::send(port.as_mut(), 0).and_then(|_| Ok(port.flush()?));
    result.and(zeroed)?;
    Ok(stats)
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
/// thread hangs up.
fn log_loop(rx: Receiver<FrameRow>, mut frames: Table) -> Result<u64> {
    let mut next_flush = Instant::now() + Duration::from_secs(1);
    loop {
        match rx.recv_timeout(next_flush.saturating_duration_since(Instant::now())) {
            Ok(r) => frames.push(r),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if Instant::now() >= next_flush {
            frames.flush()?;
            next_flush += Duration::from_secs(1);
        }
    }
    frames.finish()
}

fn quantiles(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "none".into();
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!("min {:.1}  median {:.1}  p95 {:.1}  max {:.1} ms", q(0.0), q(0.5), q(0.95), q(1.0))
}

fn report(det: &DetectStats, pol: &PolicyStats, dropped: u64, n_rows: u64, out: &std::path::Path) {
    eprintln!("\nwrote {n_rows} frames to {}", out.display());
    eprintln!("frames: {} detected, {} camera frames dropped before detection", det.frames, dropped);
    let mut ids: Vec<_> = det.with_tag.iter().collect();
    ids.sort();
    for (id, n) in ids {
        eprintln!("  tag {id}: in {n} frames ({:.1}%)", 100.0 * *n as f64 / det.frames.max(1) as f64);
    }
    eprintln!("  frame age on arrival (clock check): {}", quantiles(&mut det.age_ms.clone()));
    eprintln!("  detection: {}", quantiles(&mut det.detect_ms.clone()));
    eprintln!(
        "policy: acted on {} frames, skipped {} ({:.1}%) while busy",
        pol.acted,
        pol.skipped,
        100.0 * pol.skipped as f64 / (pol.acted + pol.skipped).max(1) as f64
    );
    eprintln!("  queued after detection: {}", quantiles(&mut pol.wait_ms.clone()));
    eprintln!("  inference: {}", quantiles(&mut pol.policy_ms.clone()));
    eprintln!("  capture to send: {}", quantiles(&mut pol.delay_ms.clone()));
}
