//! Minimal raster primitives for drawing over RGB frames.

use image::RgbImage;

pub type Color = [u8; 3];

pub const RED: Color = [235, 45, 45];
pub const GREEN: Color = [40, 220, 70];
pub const BLUE: Color = [60, 120, 255];
pub const CYAN: Color = [0, 220, 230];

/// Fills a rectangle, clipped to the image.
pub fn fill_rect(img: &mut RgbImage, x: i64, y: i64, w: i64, h: i64, color: Color) {
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
pub fn dot(img: &mut RgbImage, x: f64, y: f64, size: i64, color: Color) {
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
pub fn line(img: &mut RgbImage, a: [f64; 2], b: [f64; 2], thickness: i64, color: Color) {
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
