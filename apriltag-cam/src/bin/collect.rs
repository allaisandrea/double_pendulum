//! Records RL training data: tag poses from the camera, and the actions a
//! policy sends to the motor, on one monotonic clock.
//!
//! Two loops share a fixed action grid (slot k starts at t0 + k * period):
//!
//!   capture -> [1 slot, newest wins] -> perceive+policy -> [schedule] -> executor -> serial
//!                                              \-- rows --> logger <-- rows --/
//!
//! perceive+policy plans every --plan-every-ms: it takes a fresh frame,
//! detects tags, asks the policy for a chunk of actions, and schedules it to
//! start at a fixed offset after the frame's capture time. Planning slower
//! than the grid is what lets a chunk play out: planned every frame (~78 fps
//! here), each plan was superseded after its first action, and detection kept
//! both cores so busy that the executor stalled. The executor wakes at every slot boundary, takes the action
//! the schedule assigns to that slot, and writes it as one byte. Neither loop
//! touches the disk: the logger writes both tables.
//!
//! For now the policy is a stand-in that picks one random action per chunk.

use anyhow::{anyhow, bail, Context, Result};
use apriltag_cam::camera::{self, capture_loop, is_packed_yuyv, pick_camera, Captured};
use apriltag_cam::clock::{self, mono};
use apriltag_cam::detect::{Intrinsics, Pixels, Tracker};
use apriltag_cam::grid::{Grid, Pick, Plan, Schedule};
use apriltag_cam::mailbox::{Latest, Take};
use apriltag_cam::record::{self, ActRow, ObsRow, Table, TagRow};
use apriltag_cam::{serial, uvc};
use clap::Parser;
use nokhwa::pixel_format::RgbFormat;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;
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

    /// Grid period: one action per slot, in milliseconds
    #[arg(long, default_value_t = 20)]
    period_ms: u64,

    /// Actions per plan
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=255))]
    chunk: u16,

    /// A plan starts at the first slot at least this long after the capture
    /// time of the frame it was computed from, in milliseconds
    #[arg(long, default_value_t = 60)]
    offset_ms: u64,

    /// Plan this often, in milliseconds; 0 plans on every frame. Each plan
    /// then plays about this long before the next takes over
    #[arg(long, default_value_t = 80)]
    plan_every_ms: u64,

    /// Stand-in policy: each plan repeats one action drawn uniformly from
    /// -range..=range. 0 sends only zeros, so nothing moves
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u8).range(0..=127))]
    policy_range: u8,

    /// The stand-in policy is on for this long, then rests (all zeros) for
    /// --rest-s, and repeats, so one recording holds both driven motion and
    /// the arm settling afterwards. Seconds, on the slot grid from t0
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

    /// Fixed exposure in microseconds; 0 leaves it to the camera
    #[arg(long, default_value_t = 500)]
    exposure_us: u32,

    /// Sensor gain, 0..100
    #[arg(long, default_value_t = 100)]
    gain: u16,

    /// Tag ids to record, one column group each
    #[arg(long, value_delimiter = ',', default_value = "0,1,2")]
    tags: Vec<usize>,

    /// Edge of the tag's black square, in metres
    #[arg(long, default_value_t = 0.040)]
    tag_size: f64,

    /// Tag family
    #[arg(long, default_value = "tag36h11")]
    family: String,

    /// Horizontal field of view in degrees, when --fx/--fy are not given
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

    /// Detector threads. Two leaves the executor a core on this laptop
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

enum Row {
    Obs(ObsRow),
    Act(ActRow),
}

/// Counters both loops bump as they go, read by the periodic status line.
/// Atomics, so the executor never waits on a lock or allocates for them.
struct Live {
    frames: AtomicU64,
    /// Frames each recorded tag was seen in, in `--tags` order.
    tags: Vec<AtomicU64>,
    late_plans: AtomicU64,
    slots: AtomicU64,
    gaps: AtomicU64,
    skipped: AtomicU64,
    late_writes: AtomicU64,
    /// Worst write lateness and slowest detection since the last status
    /// line, in microseconds; the status line resets them.
    worst_write_us: AtomicU64,
    slowest_detect_us: AtomicU64,
}

impl Live {
    fn new(n_tags: usize) -> Self {
        Self {
            frames: AtomicU64::new(0),
            tags: (0..n_tags).map(|_| AtomicU64::new(0)).collect(),
            late_plans: AtomicU64::new(0),
            slots: AtomicU64::new(0),
            gaps: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            late_writes: AtomicU64::new(0),
            worst_write_us: AtomicU64::new(0),
            slowest_detect_us: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> Counts {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        Counts {
            frames: get(&self.frames),
            tags: self.tags.iter().map(get).collect(),
            late_plans: get(&self.late_plans),
            slots: get(&self.slots),
            gaps: get(&self.gaps),
            skipped: get(&self.skipped),
            late_writes: get(&self.late_writes),
        }
    }
}

#[derive(Default)]
struct Counts {
    frames: u64,
    tags: Vec<u64>,
    late_plans: u64,
    slots: u64,
    gaps: u64,
    skipped: u64,
    late_writes: u64,
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
    let us = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / 1e3;
    format!(
        "[{:5.0} s]{phase} {frames} frames, tags {}% | {} slots: {} gaps, {} skipped, \
         {} late (worst {:.1} ms) | {} late plans, slowest detection {:.0} ms",
        elapsed.as_secs_f64(),
        tags.join("/"),
        now.slots - prev.slots,
        now.gaps - prev.gaps,
        now.skipped - prev.skipped,
        now.late_writes - prev.late_writes,
        us(&live.worst_write_us),
        now.late_plans - prev.late_plans,
        us(&live.slowest_detect_us),
    )
}

/// Stand-in for a real policy: one random action, repeated for the chunk,
/// except in the slots that fall in a rest period.
struct FakePolicy {
    rng: StdRng,
    range: i8,
    chunk: usize,
    duty: DutyCycle,
}

impl FakePolicy {
    fn plan(&mut self, k_start: i64) -> Vec<i8> {
        let a = self.rng.random_range(-self.range..=self.range);
        (0..self.chunk as i64)
            .map(|i| if self.duty.resting(k_start + i) { 0 } else { a })
            .collect()
    }
}

/// Alternating active and rest periods, measured in slots from slot 0.
/// Deciding per slot, not per plan, makes a rest start exactly on time even
/// when a chunk straddles the boundary.
#[derive(Clone, Copy, Debug)]
struct DutyCycle {
    active: i64,
    rest: i64,
}

impl DutyCycle {
    fn new(active_s: f64, rest_s: f64, period: Duration) -> Self {
        let slots = |s: f64| (s / period.as_secs_f64()).round() as i64;
        Self {
            active: slots(active_s).max(1),
            rest: slots(rest_s).max(0),
        }
    }

    fn resting(&self, slot: i64) -> bool {
        self.rest > 0 && slot.rem_euclid(self.active + self.rest) >= self.active
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let period = Duration::from_millis(args.period_ms);
    let offset = Duration::from_millis(args.offset_ms);
    anyhow::ensure!(period > Duration::ZERO, "--period-ms must be positive");
    let seed = args.seed.unwrap_or_else(rand::random);
    anyhow::ensure!(args.active_s > 0.0 && args.rest_s >= 0.0, "--active-s must be positive, --rest-s not negative");
    let duty = DutyCycle::new(args.active_s, args.rest_s, period);

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

    // Slot 0 starts shortly from now, once everything is running.
    let t0 = mono() + Duration::from_millis(300);
    let grid = Grid { t0, period };
    let wall_t0 = SystemTime::now().duration_since(UNIX_EPOCH)? + (t0 - mono());

    let meta: HashMap<String, String> = [
        ("period_ns", period.as_nanos().to_string()),
        ("offset_ns", offset.as_nanos().to_string()),
        ("chunk_len", args.chunk.to_string()),
        ("plan_every_ns", Duration::from_millis(args.plan_every_ms).as_nanos().to_string()),
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
        ("policy", "fake: one uniform random action per plan, zero while resting".into()),
        ("policy_range", args.policy_range.to_string()),
        ("active_slots", duty.active.to_string()),
        ("rest_slots", duty.rest.to_string()),
        ("seed", seed.to_string()),
        ("board", serial::READY.into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    let observations = Table::create(
        &out.join("observations.arrows"),
        record::observations_schema(&args.tags, meta.clone()),
        record::observations_batch,
    )?;
    let actions = Table::create(
        &out.join("actions.arrows"),
        record::actions_schema(meta),
        record::actions_batch,
    )?;
    let (row_tx, row_rx) = mpsc::channel::<Row>();
    let logger = thread::Builder::new()
        .name("logger".into())
        .spawn(move || log_loop(row_rx, observations, actions))?;

    let schedule = Arc::new(Mutex::new(Schedule::default()));
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

    let perceive = {
        let (args, stop, schedule, rows, live) =
            (args.clone(), stop.clone(), schedule.clone(), row_tx.clone(), live.clone());
        let policy = FakePolicy {
            rng: StdRng::seed_from_u64(seed),
            range: args.policy_range as i8,
            chunk: args.chunk as usize,
            duty,
        };
        thread::Builder::new()
            .name("perceive".into())
            .spawn(move || {
                perceive_loop(&args, k, grid, offset, &slot, policy, &schedule, &rows, &live, &stop)
            })?
    };

    let executor = {
        let (stop, schedule, rows, live) =
            (stop.clone(), schedule.clone(), row_tx.clone(), live.clone());
        thread::Builder::new()
            .name("executor".into())
            .spawn(move || execute_loop(grid, port, &schedule, &rows, &live, &stop))?
    };
    drop(row_tx); // the logger finishes once both loops have dropped theirs

    let cycle = if duty.rest > 0 {
        format!("{} s on, {} s rest", args.active_s, args.rest_s)
    } else {
        "always on".to_string()
    };
    eprintln!(
        "recording to {} (seed {seed}, policy ±{}, {cycle}); Ctrl-C to stop",
        out.display(),
        args.policy_range
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
            let phase = match grid.slot_at(mono()) {
                Some(k) if duty.rest > 0 && duty.resting(k) => " resting |",
                Some(_) if duty.rest > 0 => " active  |",
                _ => "",
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
        if executor.is_finished() || perceive.is_finished() || logger.is_finished() {
            break; // a thread failed; the joins below say why
        }
        thread::sleep(Duration::from_millis(50));
    }
    stop.store(true, Ordering::SeqCst);

    // The executor first: it writes a final zero, so the motor stops before
    // anything else winds down.
    let exec = executor.join().map_err(|_| anyhow!("executor panicked"))?;
    let perc = perceive.join().map_err(|_| anyhow!("perceive thread panicked"))?;
    let logged = logger.join().map_err(|_| anyhow!("logger panicked"))?;
    let _ = watcher.join();
    let deadline = Instant::now() + Duration::from_secs(1);
    while !capture.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }

    let exec = exec.context("executor")?;
    let perc = perc.context("perception")?;
    let (n_obs, n_act) = logged.context("writing the recording")?;
    report(&exec, &perc, dropped.load(Ordering::Relaxed), n_obs, n_act, &out);
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
struct PerceiveStats {
    frames: u64,
    before_t0: u64,
    with_tag: HashMap<usize, u64>,
    detect_ms: Vec<f64>,
    /// How old each frame was when it arrived: arrival minus capture.
    age_ms: Vec<f64>,
    /// How long before its first slot each plan was ready; negative = late.
    lead_ms: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn perceive_loop(
    args: &Args,
    k: Intrinsics,
    grid: Grid,
    offset: Duration,
    slot: &Latest<Captured>,
    mut policy: FakePolicy,
    schedule: &Mutex<Schedule>,
    rows: &Sender<Row>,
    live: &Live,
    stop: &AtomicBool,
) -> Result<PerceiveStats> {
    let mut tracker = Tracker::new(&args.family, args.threads, args.decimate, args.tag_size, k)?;
    let mut stats = PerceiveStats::default();
    let ns = |t: Duration| record::rel_ns(t, grid.t0);
    let every = Duration::from_millis(args.plan_every_ms);
    let mut next_plan = grid.t0;

    while !stop.load(Ordering::Relaxed) {
        if !every.is_zero() {
            // Plan on a fixed cadence from t0; after an overrun, resume at
            // the next tick rather than catching up.
            clock::sleep_until(next_plan);
            let now = mono();
            while next_plan <= now {
                next_plan += every;
            }
        }
        // The newest frame, at most a camera period old.
        let cap = match slot.take(Duration::from_millis(100)) {
            Take::Item(cap) => cap,
            Take::Timeout => continue,
            Take::Closed => break,
        };
        let t_capture = cap
            .t_capture
            .ok_or_else(|| anyhow!("frame {} has no capture timestamp", cap.seq))?;
        if t_capture < grid.t0 {
            // Too early to record: try again at once rather than a tick later.
            stats.before_t0 += 1;
            next_plan = mono();
            continue;
        }
        stats.age_ms.push((cap.t_arrival.as_secs_f64() - t_capture.as_secs_f64()) * 1e3);

        let res = cap.buf.resolution();
        let (w, h) = (res.width() as usize, res.height() as usize);
        let started = mono();
        let tags = if is_packed_yuyv(&cap.buf) {
            tracker.detect(w, h, Pixels::Yuyv(cap.buf.buffer()))?
        } else {
            let rgb = cap.buf.decode_image::<RgbFormat>()?;
            tracker.detect(w, h, Pixels::Rgb(rgb.as_raw()))?
        };
        let t_detected = mono();
        stats.frames += 1;
        stats.detect_ms.push((t_detected - started).as_secs_f64() * 1e3);
        live.frames.fetch_add(1, Ordering::Relaxed);
        live.slowest_detect_us
            .fetch_max((t_detected - started).as_micros() as u64, Ordering::Relaxed);

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

        let t_policy = mono();
        let k_start = grid.k_start(t_capture, offset);
        let actions = policy.plan(k_start);
        let plan = Arc::new(Plan {
            frame: cap.seq,
            k_start,
            actions: actions.clone(),
        });
        schedule.lock().unwrap().push(plan);
        let t_plan = mono();
        let lead_ms = (grid.slot_time(k_start.max(0)).as_secs_f64() - t_plan.as_secs_f64()) * 1e3;
        stats.lead_ms.push(lead_ms);
        if lead_ms < 0.0 {
            live.late_plans.fetch_add(1, Ordering::Relaxed);
        }

        let _ = rows.send(Row::Obs(ObsRow {
            frame: cap.seq,
            t_capture: ns(t_capture),
            t_detected: ns(t_detected),
            tags: tag_rows,
            t_policy: Some(ns(t_policy)),
            t_plan: Some(ns(t_plan)),
            k_start: Some(k_start),
            plan: Some(actions),
        }));
    }
    Ok(stats)
}

/// Lateness histogram bin width, and how far it reaches.
const LATE_BIN: Duration = Duration::from_micros(100);
const LATE_BINS: usize = 1000;
/// Writes later than this are reported as they happen.
const LATE_WARN: Duration = Duration::from_millis(5);

struct ExecStats {
    scheduling: &'static str,
    slots: u64,
    gaps: u64,
    late: u64,
    skipped: u64,
    worst: Duration,
    /// Lateness in LATE_BIN steps; the last bin collects everything beyond.
    hist: Box<[u64; LATE_BINS + 1]>,
}

impl ExecStats {
    fn quantile(&self, q: f64) -> Duration {
        let n: u64 = self.hist.iter().sum();
        let target = (n as f64 * q).ceil() as u64;
        let mut seen = 0;
        for (i, c) in self.hist.iter().enumerate() {
            seen += c;
            if seen >= target.max(1) {
                return LATE_BIN * i as u32;
            }
        }
        Duration::ZERO
    }
}

/// Real-time budget per slot: the executor needs microseconds (a lock, a
/// one-byte write, a channel send), so 1 ms of CPU, done within 3 ms of
/// waking, is generous. A thread that overruns its budget gets demoted.
const RT_COMPUTATION: Duration = Duration::from_millis(1);
const RT_CONSTRAINT: Duration = Duration::from_millis(3);

/// Puts this thread under macOS's real-time scheduling, the policy audio
/// software uses: every `period` it is promised `RT_COMPUTATION` of CPU
/// within `RT_CONSTRAINT` of waking, ahead of ordinary threads at any
/// priority. The machine pauses for tens of milliseconds now and then, and
/// ordinary scheduling, even at the highest QoS, left the executor late
/// through them. Falls back to the highest QoS if the kernel refuses.
/// Returns which scheduling the thread got.
fn make_realtime(period: Duration) -> &'static str {
    use mach2::mach_time::{mach_timebase_info, mach_timebase_info_data_t};
    use libc::thread_policy_t;
    use mach2::thread_policy::{
        thread_policy_set, thread_time_constraint_policy_data_t, THREAD_TIME_CONSTRAINT_POLICY,
        THREAD_TIME_CONSTRAINT_POLICY_COUNT,
    };

    let mut tb = mach_timebase_info_data_t { numer: 0, denom: 0 };
    // SAFETY: `tb` is a valid, writable struct for the call to fill.
    unsafe { mach_timebase_info(&mut tb) };
    let ticks = |d: Duration| (d.as_nanos() as u64 * tb.denom as u64 / tb.numer as u64) as u32;
    let mut policy = thread_time_constraint_policy_data_t {
        period: ticks(period),
        computation: ticks(RT_COMPUTATION),
        constraint: ticks(RT_CONSTRAINT),
        preemptible: 1,
    };
    // SAFETY: the policy struct is the flavour's documented layout, and the
    // count is its size in integer_t, as thread_policy_set expects. The
    // thread's port is a send right we own, so it is released afterwards.
    let kr = unsafe {
        let thread = mach2::mach_init::mach_thread_self();
        let kr = thread_policy_set(
            thread,
            THREAD_TIME_CONSTRAINT_POLICY,
            &mut policy as *mut _ as thread_policy_t,
            THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        );
        mach2::mach_port::mach_port_deallocate(mach2::traps::mach_task_self(), thread);
        kr
    };
    if kr == 0 {
        return "real-time";
    }
    eprintln!("warning: real-time scheduling refused (kern_return {kr}); using the highest QoS");
    // SAFETY: affects only the calling thread; no pointers involved.
    let rc = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
    if rc == 0 {
        "QoS user-interactive"
    } else {
        eprintln!("warning: could not raise the executor's priority either (error {rc})");
        "default"
    }
}

fn execute_loop(
    grid: Grid,
    mut port: Box<dyn serialport::SerialPort>,
    schedule: &Mutex<Schedule>,
    rows: &Sender<Row>,
    live: &Live,
    stop: &AtomicBool,
) -> Result<ExecStats> {
    let scheduling = make_realtime(grid.period);
    let mut stats = ExecStats {
        scheduling,
        slots: 0,
        gaps: 0,
        late: 0,
        skipped: 0,
        worst: Duration::ZERO,
        hist: Box::new([0; LATE_BINS + 1]),
    };
    let mut k: i64 = 0;
    let result = (|| -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            let deadline = grid.slot_time(k);
            clock::sleep_until(deadline);
            let now = mono();
            // More than a slot behind: skip ahead rather than send a burst of
            // stale actions. The board keeps the last duty meanwhile, and the
            // skipped slots are missing from the actions table.
            if let Some(current) = grid.slot_at(now).filter(|&c| c > k) {
                stats.skipped += (current - k) as u64;
                live.skipped.fetch_add((current - k) as u64, Ordering::Relaxed);
                eprintln!("executor: skipped slots {k}..{current}, {:?} behind", now - deadline);
                k = current;
            }
            let pick = schedule.lock().unwrap().pick(k);
            serial::send(port.as_mut(), pick.action())?;
            let late = mono().saturating_sub(grid.slot_time(k));

            stats.slots += 1;
            stats.worst = stats.worst.max(late);
            live.slots.fetch_add(1, Ordering::Relaxed);
            live.worst_write_us.fetch_max(late.as_micros() as u64, Ordering::Relaxed);
            let bin = (late.as_nanos() / LATE_BIN.as_nanos()) as usize;
            stats.hist[bin.min(LATE_BINS)] += 1;
            if late > LATE_WARN {
                stats.late += 1;
                live.late_writes.fetch_add(1, Ordering::Relaxed);
                eprintln!("executor: slot {k} written {late:?} late");
            }
            let (frame, index) = match pick {
                Pick::Plan { frame, index, .. } => (Some(frame), Some(index)),
                Pick::Gap => {
                    stats.gaps += 1;
                    live.gaps.fetch_add(1, Ordering::Relaxed);
                    (None, None)
                }
            };
            let _ = rows.send(Row::Act(ActRow {
                slot: k,
                action: pick.action(),
                frame,
                index,
            }));
            k += 1;
        }
        Ok(())
    })();
    // Stop the motor whatever happened above.
    let zeroed = serial::send(port.as_mut(), 0).and_then(|_| Ok(port.flush()?));
    result.and(zeroed)?;
    Ok(stats)
}

/// Writes both tables, a batch about once a second, until both loops hang up.
fn log_loop(
    rx: Receiver<Row>,
    mut observations: Table<ObsRow>,
    mut actions: Table<ActRow>,
) -> Result<(u64, u64)> {
    let mut next_flush = Instant::now() + Duration::from_secs(1);
    loop {
        match rx.recv_timeout(next_flush.saturating_duration_since(Instant::now())) {
            Ok(Row::Obs(r)) => observations.push(r),
            Ok(Row::Act(r)) => actions.push(r),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if Instant::now() >= next_flush {
            observations.flush()?;
            actions.flush()?;
            next_flush += Duration::from_secs(1);
        }
    }
    Ok((observations.finish()?, actions.finish()?))
}

fn quantiles(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "none".into();
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!("min {:.1}  median {:.1}  p95 {:.1}  max {:.1} ms", q(0.0), q(0.5), q(0.95), q(1.0))
}

fn report(
    exec: &ExecStats,
    perc: &PerceiveStats,
    dropped: u64,
    n_obs: u64,
    n_act: u64,
    out: &std::path::Path,
) {
    let mut perc_detect = perc.detect_ms.clone();
    let mut perc_age = perc.age_ms.clone();
    let mut perc_lead = perc.lead_ms.clone();
    let late_plans = perc.lead_ms.iter().filter(|&&l| l < 0.0).count();
    eprintln!("\nwrote {n_obs} observations and {n_act} actions to {}", out.display());
    eprintln!(
        "frames: {} detected, {} camera frames dropped, {} captured before t0",
        perc.frames, dropped, perc.before_t0
    );
    let mut ids: Vec<_> = perc.with_tag.iter().collect();
    ids.sort();
    for (id, n) in ids {
        eprintln!(
            "  tag {id}: in {n} frames ({:.1}%)",
            100.0 * *n as f64 / perc.frames.max(1) as f64
        );
    }
    eprintln!("  frame age on arrival (clock check): {}", quantiles(&mut perc_age));
    eprintln!("  detection: {}", quantiles(&mut perc_detect));
    eprintln!(
        "plans: lead before first slot {}; {late_plans} late",
        quantiles(&mut perc_lead)
    );
    eprintln!(
        "executor ({}): {} slots, {} gaps, {} skipped, {} over {LATE_WARN:?} late; \
         lateness p50 {:?} p99 {:?} worst {:?}",
        exec.scheduling,
        exec.slots,
        exec.gaps,
        exec.skipped,
        exec.late,
        exec.quantile(0.5),
        exec.quantile(0.99),
        exec.worst
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rests_start_and_end_exactly_on_their_slots() {
        // 2 slots on, 3 off, at 20 ms per slot.
        let d = DutyCycle::new(0.04, 0.06, Duration::from_millis(20));
        let pattern: Vec<bool> = (0..10).map(|k| d.resting(k)).collect();
        let (f, t) = (false, true);
        assert_eq!(pattern, [f, f, t, t, t, f, f, t, t, t]);
    }

    #[test]
    fn no_rest_means_always_on() {
        let d = DutyCycle::new(1.0, 0.0, Duration::from_millis(20));
        assert!((0..1000).all(|k| !d.resting(k)));
    }

    #[test]
    fn a_chunk_straddling_a_rest_is_zeroed_from_the_boundary() {
        let mut p = FakePolicy {
            rng: StdRng::seed_from_u64(1),
            range: 30,
            chunk: 8,
            duty: DutyCycle::new(0.1, 0.1, Duration::from_millis(20)), // 5 on, 5 off
        };
        // Slots 3..11: on for 3 and 4, resting 5..=9, on again at 10.
        let plan = p.plan(3);
        let a = plan[0];
        assert_ne!(a, 0, "seed 1 draws a non-zero action");
        assert_eq!(plan, [a, a, 0, 0, 0, 0, 0, a]);
    }
}
