//! Grayscale conversion, tag detection and pose estimation.

use anyhow::{anyhow, ensure, Result};
use apriltag::{Detector, Family, Image, TagParams};

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
        let r = &self.r;
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

/// Frame pixels, in the layouts the detector can take luma from.
pub enum Pixels<'a> {
    /// Packed 4:2:2, Y0 U Y1 V: luma is every other byte, so detection
    /// needs no colour conversion at all.
    Yuyv(&'a [u8]),
    /// Packed 8-bit RGB.
    Rgb(&'a [u8]),
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

    pub fn detect(&mut self, w: usize, h: usize, pixels: Pixels) -> Result<Vec<Tag>> {
        if self
            .gray
            .as_ref()
            .is_none_or(|g| g.width() != w || g.height() != h)
        {
            self.gray = Some(Image::zeros_with_alignment(w, h, 96)?);
        }
        let gray = self.gray.as_mut().expect("allocated above");
        let stride = gray.stride();
        let buf = gray.as_slice_mut();
        match pixels {
            Pixels::Yuyv(data) => {
                ensure!(data.len() >= 2 * w * h, "YUYV frame too short for {w}x{h}");
                for (y, row) in data.chunks_exact(2 * w).take(h).enumerate() {
                    for (out, px) in buf[y * stride..][..w]
                        .iter_mut()
                        .zip(row.as_chunks::<2>().0)
                    {
                        *out = px[0];
                    }
                }
            }
            Pixels::Rgb(data) => {
                ensure!(data.len() >= 3 * w * h, "RGB frame too short for {w}x{h}");
                for (y, row) in data.chunks_exact(3 * w).take(h).enumerate() {
                    for (out, px) in buf[y * stride..][..w]
                        .iter_mut()
                        .zip(row.as_chunks::<3>().0)
                    {
                        // BT.601 luma in 8.8 fixed point.
                        let luma =
                            77 * u32::from(px[0]) + 150 * u32::from(px[1]) + 29 * u32::from(px[2]);
                        *out = (luma >> 8) as u8;
                    }
                }
            }
        }

        let tags = self
            .detector
            .detect(gray)
            .into_iter()
            .map(|d| {
                let mut solutions =
                    d.estimate_tag_pose_orthogonal_iteration(&self.params, POSE_ITERATIONS);
                solutions.sort_by(|a, b| a.error.total_cmp(&b.error));
                let pose = solutions.first().map(|s| {
                    let (r, t) = (s.pose.rotation().data(), s.pose.translation().data());
                    Pose {
                        r: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
                        t: [t[0], t[1], t[2]],
                        err: s.error,
                    }
                });
                Tag {
                    id: d.id(),
                    hamming: d.hamming(),
                    margin: d.decision_margin(),
                    pose,
                    alt_err: solutions.get(1).map(|s| s.error),
                }
            })
            .collect();
        Ok(tags)
    }
}

/// Orthogonal-iteration steps per pose; what apriltag's own
/// `estimate_tag_pose` uses.
const POSE_ITERATIONS: usize = 50;

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
            .detect(w as usize, h as usize, Pixels::Rgb(img.as_raw()))
            .unwrap();
        assert_eq!(tags.len(), 1, "expected exactly one tag");
        let tag = &tags[0];
        assert_eq!((tag.id, tag.hamming), (0, 0));

        let pose = tag.pose.expect("pose");
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
    fn yuyv_and_rgb_paths_agree() {
        let (w, h, size) = (1280u32, 720u32, 0.04);
        let k = Intrinsics::from_hfov(w, h, 60.0);
        let truth = Pose {
            r: synthetic_tag_rendering::rotation(-15.0, 30.0, -5.0),
            t: [-0.05, 0.01, 0.4],
            err: 0.0,
        };
        let img = synthetic_tag_rendering::render(w, h, &k, &truth, size);
        let yuyv: Vec<u8> = img
            .as_raw()
            .as_chunks::<6>()
            .0
            .iter()
            .flat_map(|p| [p[0], 128, p[3], 128])
            .collect();

        let mut detector = TagDetector::new("tag36h11", 2, 2.0, size, k).unwrap();
        let (w, h) = (w as usize, h as usize);
        let from_rgb = detector.detect(w, h, Pixels::Rgb(img.as_raw())).unwrap();
        let from_yuyv = detector.detect(w, h, Pixels::Yuyv(&yuyv)).unwrap();
        assert_eq!(from_rgb.len(), 1);
        assert_eq!(from_yuyv.len(), 1);
        assert_eq!(
            from_rgb[0].pose.map(|p| p.t),
            from_yuyv[0].pose.map(|p| p.t)
        );
    }

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
}
