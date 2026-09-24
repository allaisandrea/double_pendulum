//! YUYV (packed 4:2:2) to RGB conversion.
//!
//! The same BT.601 limited-range integer maths as nokhwa's decoder, but as a
//! flat loop into a reused buffer: several times faster, which matters with
//! two cores shared between detection and encoding.

/// Converts packed Y0 U Y1 V bytes into RGB, 3 bytes per pixel.
pub fn to_rgb(yuyv: &[u8], rgb: &mut [u8]) {
    for (src, dst) in yuyv
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<6>().0)
    {
        let [y0, u, y1, v] = src.map(i32::from);
        let (d, e) = (u - 128, v - 128);
        let (r, g, b) = (409 * e + 128, -100 * d - 208 * e + 128, 516 * d + 128);
        for (px, y) in dst.as_chunks_mut::<3>().0.iter_mut().zip([y0, y1]) {
            let c = (y - 16) * 298;
            *px = [(c + r) >> 8, (c + g) >> 8, (c + b) >> 8].map(|x| x.clamp(0, 255) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_nokhwa_byte_for_byte() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let yuyv: Vec<u8> = (0..64 * 48 * 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        let want = nokhwa::utils::yuyv422_to_rgb(&yuyv, false).unwrap();
        let mut got = vec![0; want.len()];
        super::to_rgb(&yuyv, &mut got);
        assert_eq!(got, want);
    }
}
