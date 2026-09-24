//! Facts about the rig that the programs share.

/// Edge of each tag's black square, in metres: 0.8x the printed sheet's
/// nominal size.
pub const TAG_SIZE_M: f64 = 0.023;

/// The AprilTag family the tags are printed in.
pub const TAG_FAMILY: &str = "tag36h11";

/// The Arducam's horizontal field of view in degrees, giving the nominal
/// intrinsics: square pixels, principal point at the image centre. Estimated
/// from the camera's measured distance to tag 0 (see the README); it is not
/// a calibration.
pub const HFOV_DEG: f64 = 72.0;
