//! Draws each detected tag's pose over a frame.

use crate::detect::{Intrinsics, Tag};
use crate::draw::{self, BLUE, CYAN, GREEN, RED};
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
                draw::line(img, a, b, s, CYAN);
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
                draw::line(img, origin, tip, 2 * s, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::{Pixels, Pose, Tracker};
    use crate::synthetic;

    /// Draws the overlay on a synthetic frame. Set TAGCAM_DUMP=<path.jpg> to
    /// write the result out for a visual check.
    #[test]
    fn overlay_draws_the_pose() {
        let (w, h, size) = (1280, 720, 0.04);
        let k = Intrinsics::from_hfov(w, h, 60.0);
        let truth = Pose {
            r: synthetic::rotation(25.0, -20.0, 10.0),
            t: [0.03, -0.02, 0.35],
            err: 0.0,
        };
        let mut img = synthetic::render(w, h, &k, &truth, size);
        let tags = Tracker::new("tag36h11", 2, 2.0, size, k)
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

        if let Ok(path) = std::env::var("TAGCAM_DUMP") {
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
