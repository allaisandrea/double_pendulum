//! Sets a camera's exposure through UVC, by driving the vendored `uvc-util`.
//!
//! AVFoundation exposes no exposure-time control for external cameras on
//! macOS, so nokhwa cannot do this: its `set_control` goes through
//! `setExposureModeCustomWithDuration:ISO:`, which those cameras ignore.
//! UVC requests are the only route, and `tools/uvc-util` already speaks them.
//!
//! Two properties of these controls drive the design here:
//!
//!   * **A rejected write looks exactly like a successful one.** While the
//!     camera is in auto mode it ignores writes to the exposure time,
//!     reports success, and will even read back the value it was given while
//!     continuing to meter automatically. So every write is verified.
//!   * **The settings are volatile.** They live in the camera, not the host,
//!     and a USB re-enumeration silently restores the defaults. That has
//!     happened mid-session on this rig, so the values are checked again
//!     after recording and a change is reported rather than passed over.

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Mode 1 is manual exposure; 8 is the aperture-priority auto mode these
/// cameras come up in.
const MODE_MANUAL: &str = "1";

/// `exposure-time-abs` counts in units of 100 us, per the UVC spec.
const EXPOSURE_UNIT_US: u32 = 100;

/// What was asked for, and what the camera actually accepted.
pub struct Applied {
    pub exposure_us: u32,
    pub gain: Option<u16>,
}

/// The camera as `uvc-util` sees it: its own index, which is unrelated to
/// both nokhwa's index and the order cameras are listed in.
pub struct Device {
    tool: PathBuf,
    index: String,
    pub name: String,
}

impl Device {
    /// Finds the UVC device behind an AVFoundation unique id.
    ///
    /// The unique id is one hex number packing the USB location id, vendor
    /// and product, all of which `uvc-util -d` prints, so the two can be
    /// matched without relying on device names that may repeat. Compare them
    /// as numbers: the unique id drops the location's leading zeros, so
    /// location 0x01110000 appears as `0x1110000...`.
    pub fn find(camera_id: &str) -> Result<Self> {
        let tool = find_tool()?;
        let listing = run(&tool, &["-d"])?;
        let camera = parse_hex(camera_id);
        let mut candidates = Vec::new();
        for line in listing.lines() {
            // index, vend:prod, location, uvc version, then the name
            let mut f = line.split_whitespace();
            let (Some(index), Some(vp), Some(location), Some(_ver)) =
                (f.next(), f.next(), f.next(), f.next())
            else {
                continue;
            };
            if index.parse::<u32>().is_err() || !location.starts_with("0x") {
                continue; // a header or separator row
            }
            let name = f.collect::<Vec<_>>().join(" ");
            let unique_id = (|| {
                let (vendor, product) = vp.split_once(':')?;
                Some(parse_hex(location)? << 32 | parse_hex(vendor)? << 16 | parse_hex(product)?)
            })();
            if camera.is_some() && camera == unique_id {
                return Ok(Self {
                    tool,
                    index: index.to_string(),
                    name,
                });
            }
            candidates.push(format!("{location} ({name})"));
        }
        bail!(
            "no UVC device matches camera {camera_id}; uvc-util sees: {}",
            if candidates.is_empty() {
                "none".to_string()
            } else {
                candidates.join(", ")
            }
        )
    }

    fn get(&self, control: &str) -> Result<String> {
        let out = run(&self.tool, &["-I", &self.index, "-S", control])?;
        out.lines()
            .find_map(|l| l.trim().strip_prefix("current-value:"))
            .map(|v| v.trim().to_string())
            .ok_or_else(|| anyhow!("uvc-util reported no current-value for {control}"))
    }

    /// Writes one control and reads it back. Each control goes in its own
    /// invocation: batching several `-s` flags into one call proved
    /// unreliable, with later writes silently dropped.
    fn set(&self, control: &str, value: &str) -> Result<()> {
        run(&self.tool, &["-I", &self.index, "-s", &format!("{control}={value}")])
            .with_context(|| format!("setting {control}={value}"))?;
        let got = self.get(control)?;
        if got != value {
            bail!("{control} would not take {value}; it reads {got}");
        }
        Ok(())
    }

    /// Puts the camera in manual exposure and applies the given values.
    ///
    /// The mode goes first and is verified before anything else, because in
    /// auto mode the writes that follow would be accepted and ignored.
    pub fn set_exposure(&self, exposure_us: u32, gain: Option<u16>) -> Result<Applied> {
        self.set("auto-exposure-mode", MODE_MANUAL)
            .context("camera would not leave auto exposure")?;

        let units = (exposure_us + EXPOSURE_UNIT_US / 2) / EXPOSURE_UNIT_US;
        let units = units.max(1);
        self.set("exposure-time-abs", &units.to_string())?;
        if let Some(g) = gain {
            self.set("gain", &g.to_string())?;
        }
        Ok(Applied {
            exposure_us: units * EXPOSURE_UNIT_US,
            gain,
        })
    }

    /// Re-reads the controls and describes any drift from what was applied.
    /// Returns None while everything still matches.
    pub fn drift(&self, applied: &Applied) -> Option<String> {
        let mode = self.get("auto-exposure-mode").ok()?;
        if mode != MODE_MANUAL {
            return Some(format!(
                "the camera returned to automatic exposure (mode {mode}); \
                 it most likely dropped off the USB bus and re-enumerated"
            ));
        }
        let units = (applied.exposure_us / EXPOSURE_UNIT_US).to_string();
        match self.get("exposure-time-abs") {
            Ok(v) if v != units => {
                return Some(format!("exposure changed from {units} to {v} (units of 100 us)"))
            }
            _ => {}
        }
        if let Some(g) = applied.gain {
            match self.get("gain") {
                Ok(v) if v != g.to_string() => {
                    return Some(format!("gain changed from {g} to {v}"))
                }
                _ => {}
            }
        }
        None
    }
}

/// Looks for `uvc-util`: an explicit override first, then the copy this
/// repository builds, then whatever is on PATH.
fn find_tool() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("UVC_UTIL") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        bail!("UVC_UTIL is set to {}, which is not a file", p.display());
    }
    // src/ -> apriltag-cam/ -> the repository root
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent();
    if let Some(p) = repo.map(|r| r.join("tools/bin/uvc-util")).filter(|p| p.is_file()) {
        return Ok(p);
    }
    if let Ok(p) = which("uvc-util") {
        return Ok(p);
    }
    bail!(
        "uvc-util not found; build it with `make -C tools`, or point UVC_UTIL at a copy. \
         It is needed because --exposure-us cannot be set through AVFoundation."
    )
}

fn which(name: &str) -> Result<PathBuf> {
    let path = std::env::var_os("PATH").ok_or_else(|| anyhow!("PATH is not set"))?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| anyhow!("{name} is not on PATH"))
}

fn run(tool: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new(tool)
        .args(args)
        .output()
        .with_context(|| format!("running {}", tool.display()))?;
    if !out.status.success() {
        bail!(
            "{} {} failed: {}",
            tool.display(),
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn parse_hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.strip_prefix("0x")?, 16).ok()
}
