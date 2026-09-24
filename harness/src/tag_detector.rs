//! Grayscale conversion, tag detection and pose estimation.

use anyhow::{anyhow, ensure, Result};
use apriltag::{Detection, Detector, Family, Image, PoseEstimation, TagParams};

/// Pinhole camera intrinsics, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Intrinsics {
    /// Square pixels, principal point at the image centre and the focal
    /// length implied by the horizontal field of view. A stand-in until the
    /// camera is calibrated: distances scale with the error in `fx`.
    pub fn from_hfov(width: u32, height: u32, hfov_deg: f64) -> Self {
        let fx = f64::from(width) / 2.0 / (hfov_deg.to_radians() / 2.0).tan();
        Self {
            fx,
            fy: fx,
            cx: f64::from(width) / 2.0,
            cy: f64::from(height) / 2.0,
        }
    }

    /// Projects a camera-frame point to pixels; `None` behind the camera.
    pub(crate) fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        (p[2] > 1e-6).then(|| {
            [
                self.fx * p[0] / p[2] + self.cx,
                self.fy * p[1] / p[2] + self.cy,
            ]
        })
    }
}

/// A tag's pose in the camera frame (x right, y down, z out of the lens).
/// The tag frame has its origin at the tag centre, x right, y down and z
/// pointing into the tag.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    /// Rotation taking tag-frame vectors into the camera frame, row-major.
    pub(crate) r: [[f64; 3]; 3],
    /// Tag centre in the camera frame, metres.
    pub t: [f64; 3],
    /// Object-space error of this solution.
    pub err: f64,
}

impl Pose {
    /// Maps a tag-frame point into the camera frame.
    pub(crate) fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.r;
        std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + self.t[i])
    }

    /// The rotation as a unit quaternion (w, x, y, z) with w >= 0.
    pub fn quaternion(&self) -> [f64; 4] {
        quaternion_from_rotation(&self.r)
    }
}

/// One detected tag.
#[derive(Clone, Debug)]
pub struct Tag {
    pub id: usize,
    pub hamming: usize,
    pub margin: f32,
    /// The lower-error pose solution.
    pub pose: Option<Pose>,
    /// Error of the competing solution. A planar tag has two poses that fit
    /// its corners; when this approaches `pose.err` the two are nearly
    /// indistinguishable and the pose can flip between frames.
    pub alt_err: Option<f64>,
}

pub struct TagDetector {
    detector: Detector,
    /// Reused between frames; apriltag wants its own stride-aligned buffer.
    gray: Option<Image>,
    params: TagParams,
}

impl TagDetector {
    pub fn new(
        family: &str,
        threads: u8,
        decimate: f32,
        tag_size: f64,
        k: Intrinsics,
    ) -> Result<Self> {
        let family: Family = family
            .parse()
            .map_err(|e| anyhow!("unknown tag family {family:?}: {e:?}"))?;
        let mut detector = Detector::builder().add_family_bits(family, 1).build()?;
        detector.set_thread_number(threads);
        detector.set_decimation(decimate);
        let params = TagParams {
            tagsize: tag_size,
            fx: k.fx,
            fy: k.fy,
            cx: k.cx,
            cy: k.cy,
        };
        Ok(Self {
            detector,
            gray: None,
            params,
        })
    }

    /// Detects tags in a `w` x `h` frame of packed 4:2:2 YUYV (Y0 U Y1 V):
    /// luma is every other byte, so detection needs no colour conversion.
    pub fn detect(&mut self, w: usize, h: usize, yuyv: &[u8]) -> Result<Vec<Tag>> {
        let gray = gray_image(&mut self.gray, w, h)?;
        luma_from_yuyv(yuyv, gray)?;
        let detections = self.detector.detect(gray);
        Ok(detections.iter().map(|d| to_tag(d, &self.params)).collect())
    }
}

/// Orthogonal-iteration steps per pose; what apriltag's own
/// `estimate_tag_pose` uses.
const POSE_ITERATIONS: usize = 50;

/// The grayscale image in `slot`, allocated on first use and again whenever
/// the frame size changes.
fn gray_image(slot: &mut Option<Image>, w: usize, h: usize) -> Result<&mut Image> {
    if slot
        .as_ref()
        .is_none_or(|g| g.width() != w || g.height() != h)
    {
        *slot = Some(Image::zeros_with_alignment(w, h, 96)?);
    }
    Ok(slot.as_mut().expect("allocated above"))
}

/// Copies the luma of a packed YUYV frame into `gray`: every other byte.
/// Packed YUYV is exactly 2 bytes a pixel, so a frame of any other length is
/// in some other layout (NV12, MJPEG) and is refused rather than misread.
fn luma_from_yuyv(data: &[u8], gray: &mut Image) -> Result<()> {
    let (w, h, stride) = (gray.width(), gray.height(), gray.stride());
    ensure!(
        data.len() == 2 * w * h,
        "frame is {} bytes, but packed YUYV at {w}x{h} is {}",
        data.len(),
        2 * w * h
    );
    let buf = gray.as_slice_mut();
    for (y, row) in data.chunks_exact(2 * w).take(h).enumerate() {
        for (out, px) in buf[y * stride..][..w]
            .iter_mut()
            .zip(row.as_chunks::<2>().0)
        {
            *out = px[0];
        }
    }
    Ok(())
}

/// Estimates a detection's pose and describes it as a [`Tag`]: the better of
/// the two pose solutions, and the error of the other.
fn to_tag(d: &Detection, params: &TagParams) -> Tag {
    let mut solutions = d.estimate_tag_pose_orthogonal_iteration(params, POSE_ITERATIONS);
    solutions.sort_by(|a, b| a.error.total_cmp(&b.error));
    Tag {
        id: d.id(),
        hamming: d.hamming(),
        margin: d.decision_margin(),
        pose: solutions.first().map(to_pose),
        alt_err: solutions.get(1).map(|s| s.error),
    }
}

/// One of apriltag's pose solutions, as a [`Pose`].
fn to_pose(s: &PoseEstimation) -> Pose {
    let (r, t) = (s.pose.rotation().data(), s.pose.translation().data());
    Pose {
        r: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
        t: [t[0], t[1], t[2]],
        err: s.error,
    }
}

/// A rotation matrix as a unit quaternion (w, x, y, z) with w >= 0. q and
/// -q are the same rotation; picking w >= 0 makes the result unique.
/// Branches on the largest diagonal term, so the square root never takes a
/// small argument.
fn quaternion_from_rotation(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let q = if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).sqrt();
        [
            s / 4.0,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2.0 * (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt();
        [
            (r[2][1] - r[1][2]) / s,
            s / 4.0,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = 2.0 * (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt();
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            s / 4.0,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt();
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            s / 4.0,
        ]
    };
    if q[0] < 0.0 {
        q.map(|v| -v)
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic_tag_rendering;

    fn angle_deg(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
        // trace(a^T b) = 1 + 2 cos(angle)
        let trace: f64 = (0..3)
            .map(|i| (0..3).map(|j| a[j][i] * b[j][i]).sum::<f64>())
            .sum();
        ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Renders tag36h11 id 0 exactly as make_apriltag_sheet.py prints it, at
    /// a known oblique pose, and checks the detector decodes it and recovers
    /// that pose: this pins down the bit layout of the printed sheet, the
    /// tag-frame axis convention and the rotation/translation layout.
    #[test]
    fn recovers_a_known_pose() {
        let (w, h, size) = (1280, 720, 0.04);
        let k = Intrinsics::from_hfov(w, h, 60.0);
        let truth = Pose {
            r: synthetic_tag_rendering::rotation(25.0, -20.0, 10.0),
            t: [0.03, -0.02, 0.35],
            err: 0.0,
        };
        let img = synthetic_tag_rendering::render(w, h, &k, &truth, size);

        let tags = TagDetector::new("tag36h11", 2, 2.0, size, k)
            .unwrap()
            .detect(
                w as usize,
                h as usize,
                &synthetic_tag_rendering::to_yuyv(&img),
            )
            .unwrap();
        assert_eq!(tags.len(), 1, "expected exactly one tag");
        let tag = &tags[0];
        assert_eq!((tag.id, tag.hamming), (0, 0));

        let pose = tag.pose.expect("pose");
        // The pose is the better of the two solutions; this oblique view has
        // a second one.
        let alt_err = tag.alt_err.expect("a second pose solution");
        assert!(alt_err > pose.err, "alt_err {alt_err} <= err {}", pose.err);
        let dt = (0..3)
            .map(|i| (pose.t[i] - truth.t[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            dt < 0.005,
            "translation off by {:.1} mm: {:?}",
            dt * 1e3,
            pose.t
        );
        let angle = angle_deg(&pose.r, &truth.r);
        assert!(angle < 2.0, "rotation off by {angle:.2} deg: {:?}", pose.r);
    }

    /// The YUYV path must see exactly the luma the RGB path computes: for a
    /// grey image both are the grey level itself.
    #[test]
    fn quaternion_of_a_quarter_turn_about_z() {
        let pose = Pose {
            r: synthetic_tag_rendering::rotation(0.0, 0.0, 90.0),
            t: [0.0; 3],
            err: 0.0,
        };
        let h = 0.5_f64.sqrt();
        for (got, want) in pose.quaternion().iter().zip([h, 0.0, 0.0, h]) {
            assert!((got - want).abs() < 1e-9, "{:?}", pose.quaternion());
        }
    }

    /// The rotation matrix of a unit quaternion (w, x, y, z).
    fn rotation_from_quaternion([w, x, y, z]: [f64; 4]) -> [[f64; 3]; 3] {
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    #[test]
    fn quaternion_round_trips_through_every_branch() {
        // Half turns about each axis land in the three branches the trace
        // test skips; the grid covers the rest.
        let half_turns = [(180.0, 0.0, 0.0), (0.0, 180.0, 0.0), (0.0, 0.0, 180.0)];
        let steps = [-170.0, -120.0, -60.0, 0.0, 45.0, 100.0, 150.0];
        let mut angles = half_turns.to_vec();
        for a in steps {
            for b in steps {
                for c in steps {
                    angles.push((a, b, c));
                }
            }
        }
        for (a, b, c) in angles {
            let r = synthetic_tag_rendering::rotation(a, b, c);
            let q = quaternion_from_rotation(&r);
            let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-9,
                "|q| = {norm} for ({a}, {b}, {c})"
            );
            assert!(q[0] >= 0.0, "w < 0 for ({a}, {b}, {c}): {q:?}");
            // Element-wise: angle_deg's acos cannot resolve angles this small.
            let back = rotation_from_quaternion(q);
            let diff = (0..9)
                .map(|i| (r[i / 3][i % 3] - back[i / 3][i % 3]).abs())
                .fold(0.0, f64::max);
            assert!(diff < 1e-12, "({a}, {b}, {c}) came back {diff} off");
        }
        // A half turn about x is (0, 1, 0, 0), up to sign.
        let q = quaternion_from_rotation(&synthetic_tag_rendering::rotation(180.0, 0.0, 0.0));
        assert!((q[1].abs() - 1.0).abs() < 1e-9, "{q:?}");
    }

    #[test]
    fn the_gray_image_follows_the_frame_size() {
        let mut slot = None;
        let first = gray_image(&mut slot, 64, 48).unwrap();
        assert_eq!((first.width(), first.height()), (64, 48));
        let ptr = first.as_slice_mut().as_ptr();
        // The same size reuses the buffer; a new size replaces it.
        let same = gray_image(&mut slot, 64, 48).unwrap();
        assert_eq!(same.as_slice_mut().as_ptr(), ptr);
        let resized = gray_image(&mut slot, 100, 30).unwrap();
        assert_eq!((resized.width(), resized.height()), (100, 30));
    }
}
