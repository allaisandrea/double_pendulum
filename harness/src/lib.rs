//! The laptop side of the double pendulum rig: camera capture, AprilTag pose
//! estimation, the policy interface and the motor link, shared by the
//! binaries `record`, which records annotated video, and `collect`, which
//! records RL training data while driving the motor.

pub mod camera;
pub mod clock;
pub mod detect;
mod draw;
pub mod latest;
pub mod mov;
pub mod overlay_tag;
pub mod policy;
pub mod record;
pub mod serial;
#[cfg(test)]
mod synthetic_tag_rendering;
pub mod uvc;
pub mod yuyv;
