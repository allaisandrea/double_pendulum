//! Renders a tag at a known pose, for tests.

use crate::tag_detector::{Intrinsics, Pose};
use image::RgbImage;

/// tag36h11 id 0 as make_apriltag_sheet.py prints it: the 6x6 data bits
/// row-major from the top-left, most significant first, a set bit white.
const TAG0: u64 = 0xD5D628584;

/// Grey level of the module at (col, row), counted in modules from the
/// black square's top-left corner; `None` beyond the one-module white border.
fn module(col: i64, row: i64) -> Option<u8> {
    if !(-1..9).contains(&col) || !(-1..9).contains(&row) {
        return None;
    }
    if !(0..8).contains(&col) || !(0..8).contains(&row) {
        return Some(255);
    }
    if col == 0 || col == 7 || row == 0 || row == 7 {
        return Some(0);
    }
    let bit = (row - 1) * 6 + (col - 1);
    Some(if (TAG0 >> (35 - bit)) & 1 == 1 {
        255
    } else {
        0
    })
}

/// Rotation Rx(ax) Ry(ay) Rz(az), angles in degrees.
pub fn rotation(ax: f64, ay: f64, az: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = ax.to_radians().sin_cos();
    let (sb, cb) = ay.to_radians().sin_cos();
    let (sc, cc) = az.to_radians().sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]];
    let ry = [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]];
    let rz = [[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]];
    mul(&mul(&rx, &ry), &rz)
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Ray-traces the tag, black square `tag_size` metres across, posed by
/// `pose` (tag frame: centre origin, x right, y down, z into the tag), over
/// a mid-grey background, with 3x3 supersampling.
pub fn render(width: u32, height: u32, k: &Intrinsics, pose: &Pose, tag_size: f64) -> RgbImage {
    const SS: u32 = 3;
    let (r, t) = (&pose.r, &pose.t);
    let normal = [r[0][2], r[1][2], r[2][2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let n_dot_t = dot(normal, *t);
    let pitch = tag_size / 8.0;

    RgbImage::from_fn(width, height, |u, v| {
        let mut sum = 0u32;
        for sy in 0..SS {
            for sx in 0..SS {
                let px = f64::from(u) + (f64::from(sx) + 0.5) / f64::from(SS);
                let py = f64::from(v) + (f64::from(sy) + 0.5) / f64::from(SS);
                let ray = [(px - k.cx) / k.fx, (py - k.cy) / k.fy, 1.0];
                let depth = n_dot_t / dot(normal, ray);
                let grey = if depth > 0.0 {
                    let p: [f64; 3] = std::array::from_fn(|i| depth * ray[i] - t[i]);
                    // Tag-frame coordinates: R^T p.
                    let x = r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2];
                    let y = r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2];
                    let col = ((x + tag_size / 2.0) / pitch).floor() as i64;
                    let row = ((y + tag_size / 2.0) / pitch).floor() as i64;
                    module(col, row).unwrap_or(128)
                } else {
                    128
                };
                sum += u32::from(grey);
            }
        }
        let g = (sum / (SS * SS)) as u8;
        image::Rgb([g, g, g])
    })
}
