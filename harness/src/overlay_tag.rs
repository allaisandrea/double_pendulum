//! Draws each detected tag's pose over a frame, with a few minimal raster
//! primitives for drawing over RGB frames.

use crate::tag_detector::{Intrinsics, Tag};
use image::RgbImage;

/// Draws the tag's axes and a cube standing on the tag. A cube that stays
/// square and flush on the tag through motion is the quickest visual check
/// that the pose is right. Tags without a pose are not drawn.
pub fn draw_tag(img: &mut RgbImage, tag: &Tag, k: &Intrinsics, tag_size: f64) {
    let Some(pose) = &tag.pose else {
        return;
    };
    // Line width: 1 px per 640 px of frame width.
    let s = (i64::from(img.width()) / 640).max(1);
    let project = |p: [f64; 3]| k.project(pose.apply(p));

    let h = tag_size / 2.0;
    let base = [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]];
    // +z points into the tag, so the cube rises along -z.
    let top = base.map(|[x, y, _]| [x, y, -tag_size]);
    for i in 0..4 {
        let j = (i + 1) % 4;
        for (a, b) in [(base[i], base[j]), (top[i], top[j]), (base[i], top[i])] {
            if let (Some(a), Some(b)) = (project(a), project(b)) {
                line(img, a, b, s, CYAN);
            }
        }
    }

    let len = 0.75 * tag_size;
    if let Some(origin) = project([0.0; 3]) {
        for (tip, color) in [
            ([len, 0.0, 0.0], RED),
            ([0.0, len, 0.0], GREEN),
            ([0.0, 0.0, len], BLUE),
        ] {
            if let Some(tip) = project(tip) {
                line(img, origin, tip, 2 * s, color);
            }
        }
    }
}

type Color = [u8; 3];

const RED: Color = [235, 45, 45];
const GREEN: Color = [40, 220, 70];
const BLUE: Color = [60, 120, 255];
const CYAN: Color = [0, 220, 230];

/// Fills a rectangle, clipped to the image.
fn fill_rect(img: &mut RgbImage, x: i64, y: i64, w: i64, h: i64, color: Color) {
    let (x0, y0) = (x.max(0), y.max(0));
    let x1 = (x + w).min(i64::from(img.width()));
    let y1 = (y + h).min(i64::from(img.height()));
    for yy in y0..y1 {
        for xx in x0..x1 {
            img.put_pixel(xx as u32, yy as u32, image::Rgb(color));
        }
    }
}

/// A filled square of side `size` centred on (x, y).
fn dot(img: &mut RgbImage, x: f64, y: f64, size: i64, color: Color) {
    let half = size as f64 / 2.0;
    fill_rect(
        img,
        (x - half).round() as i64,
        (y - half).round() as i64,
        size,
        size,
        color,
    );
}

/// A line `thickness` pixels wide. Segments are clipped to the image first,
/// so endpoints far off-screen (a pose projected near the camera plane)
/// cost nothing.
fn line(img: &mut RgbImage, a: [f64; 2], b: [f64; 2], thickness: i64, color: Color) {
    let (w, h) = (f64::from(img.width()) - 1.0, f64::from(img.height()) - 1.0);
    let Some((a, b)) = clip(a, b, w, h) else {
        return;
    };
    let steps = (b[0] - a[0]).hypot(b[1] - a[1]).ceil().max(1.0) as usize;
    for i in 0..=steps {
        let s = i as f64 / steps as f64;
        dot(
            img,
            a[0] + s * (b[0] - a[0]),
            a[1] + s * (b[1] - a[1]),
            thickness,
            color,
        );
    }
}

/// Liang-Barsky clip of segment a-b to [0, w] x [0, h].
fn clip(a: [f64; 2], b: [f64; 2], w: f64, h: f64) -> Option<([f64; 2], [f64; 2])> {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return None;
    }
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, q) in [(-dx, a[0]), (dx, w - a[0]), (-dy, a[1]), (dy, h - a[1])] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((
        [a[0] + t0 * dx, a[1] + t0 * dy],
        [a[0] + t1 * dx, a[1] + t1 * dy],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic_tag_rendering;
    use crate::tag_detector::{Pixels, Pose, TagDetector};

    /// Draws the overlay on a synthetic frame. Set OVERLAY_DUMP=<path.jpg> to
    /// write the result out for a visual check.
    #[test]
    fn overlay_draws_the_pose() {
        let (w, h, size) = (1280, 720, 0.04);
        let k = Intrinsics::from_hfov(w, h, 60.0);
        let truth = Pose {
            r: synthetic_tag_rendering::rotation(25.0, -20.0, 10.0),
            t: [0.03, -0.02, 0.35],
            err: 0.0,
        };
        let mut img = synthetic_tag_rendering::render(w, h, &k, &truth, size);
        let tags = TagDetector::new("tag36h11", 2, 2.0, size, k)
            .unwrap()
            .detect(w as usize, h as usize, Pixels::Rgb(img.as_raw()))
            .unwrap();
        for tag in &tags {
            draw_tag(&mut img, tag, &k, size);
        }

        // The axes meet at the tag centre, and z is drawn last.
        let pose = tags[0].pose.expect("pose");
        let [x, y] = k.project(pose.apply([0.0; 3])).unwrap();
        assert_eq!(img.get_pixel(x as u32, y as u32).0, BLUE);

        if let Ok(path) = std::env::var("OVERLAY_DUMP") {
            let mut jpeg = Vec::new();
            jpeg_encoder::Encoder::new(&mut jpeg, 90)
                .encode(
                    img.as_raw(),
                    w as u16,
                    h as u16,
                    jpeg_encoder::ColorType::Rgb,
                )
                .unwrap();
            std::fs::write(path, jpeg).unwrap();
        }
    }
}
