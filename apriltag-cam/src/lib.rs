//! Camera capture and AprilTag pose estimation shared by the binaries:
//! `apriltag-cam`, which records annotated video, and `collect`, which
//! records RL training data while driving the motor.

pub mod camera;
pub mod clock;
pub mod detect;
pub mod draw;
pub mod grid;
pub mod mailbox;
pub mod mov;
pub mod overlay;
pub mod record;
pub mod serial;
#[cfg(test)]
mod synthetic;
pub mod uvc;
pub mod yuyv;
