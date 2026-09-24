//! Minimal QuickTime (.mov) writer for Motion-JPEG video.
//!
//! Every frame is a standalone JPEG, so QuickTime Player can step through a
//! recording frame by frame with no inter-frame smearing of tag edges.
//! Frames are appended to `mdat` as they arrive and the sample tables are
//! written into `moov` by [`MovWriter::finish`]; a file that is never
//! finished has no index and will not play.

use anyhow::{ensure, Result};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct MovWriter {
    file: BufWriter<File>,
    width: u16,
    height: u16,
    mdat_start: u64,
    end: u64,
    offsets: Vec<u64>,
    sizes: Vec<u32>,
    pts: Vec<Duration>,
}

impl MovWriter {
    pub fn create(path: &Path, width: u16, height: u16) -> Result<Self> {
        let mut file = BufWriter::new(File::create(path)?);
        let mut ftyp = Atom::new(b"ftyp");
        ftyp.bytes(b"qt  ");
        ftyp.u32(0x0000_0200);
        ftyp.bytes(b"qt  ");
        let ftyp = ftyp.finish();
        file.write_all(&ftyp)?;

        // 64-bit mdat header (size field 1 means "see largesize"), so the
        // recording can pass 4 GB; the real size is patched in by finish().
        let mdat_start = ftyp.len() as u64;
        file.write_all(&1u32.to_be_bytes())?;
        file.write_all(b"mdat")?;
        file.write_all(&0u64.to_be_bytes())?;

        Ok(Self {
            file,
            width,
            height,
            mdat_start,
            end: mdat_start + 16,
            offsets: Vec::new(),
            sizes: Vec::new(),
            pts: Vec::new(),
        })
    }

    /// Appends one JPEG frame presented at `pts`, the time since recording
    /// started. Real capture times give real-time playback even when frames
    /// were dropped.
    pub fn write_frame(&mut self, jpeg: &[u8], pts: Duration) -> Result<()> {
        self.file.write_all(jpeg)?;
        self.offsets.push(self.end);
        self.sizes.push(jpeg.len().try_into()?);
        self.pts.push(pts);
        self.end += jpeg.len() as u64;
        Ok(())
    }

    /// Writes the sample index and patches the mdat size.
    pub fn finish(mut self) -> Result<()> {
        ensure!(!self.offsets.is_empty(), "no frames were recorded");
        let moov = self.moov();
        self.file.write_all(&moov)?;
        self.file.seek(SeekFrom::Start(self.mdat_start + 8))?;
        self.file
            .write_all(&(self.end - self.mdat_start).to_be_bytes())?;
        self.file.flush()?;
        Ok(())
    }

    /// Per-frame durations in media ticks, from consecutive timestamps. The
    /// last frame repeats the previous interval, or 1/30 s if it is alone.
    fn durations(&self) -> Vec<u32> {
        let ticks = |d: &Duration| (d.as_secs_f64() * f64::from(MEDIA_TIMESCALE)).round() as i64;
        let mut out: Vec<u32> = self
            .pts
            .windows(2)
            .map(|w| (ticks(&w[1]) - ticks(&w[0])).max(1) as u32)
            .collect();
        out.push(out.last().copied().unwrap_or(MEDIA_TIMESCALE / 30));
        out
    }

    fn moov(&self) -> Vec<u8> {
        let durations = self.durations();
        let media_duration: u64 = durations.iter().map(|&d| u64::from(d)).sum();
        let movie_duration =
            (media_duration * u64::from(MOVIE_TIMESCALE) / u64::from(MEDIA_TIMESCALE)) as u32;
        let now = (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
            + QT_EPOCH_OFFSET) as u32;

        let mut mvhd = Atom::full(b"mvhd", 0, 0);
        mvhd.u32(now);
        mvhd.u32(now);
        mvhd.u32(MOVIE_TIMESCALE);
        mvhd.u32(movie_duration);
        mvhd.u32(0x0001_0000); // rate 1.0
        mvhd.u16(0x0100); // volume 1.0
        mvhd.zeros(10);
        mvhd.matrix();
        mvhd.zeros(24); // preview, poster, selection and current times
        mvhd.u32(2); // next track id

        let mut tkhd = Atom::full(b"tkhd", 0, 0x3); // enabled, in movie
        tkhd.u32(now);
        tkhd.u32(now);
        tkhd.u32(1); // track id
        tkhd.zeros(4);
        tkhd.u32(movie_duration);
        tkhd.zeros(8);
        tkhd.zeros(6); // layer, alternate group, volume (none for video)
        tkhd.zeros(2);
        tkhd.matrix();
        tkhd.u32(u32::from(self.width) << 16);
        tkhd.u32(u32::from(self.height) << 16);

        let mut mdhd = Atom::full(b"mdhd", 0, 0);
        mdhd.u32(now);
        mdhd.u32(now);
        mdhd.u32(MEDIA_TIMESCALE);
        mdhd.u32(media_duration as u32);
        mdhd.u16(0x55c4); // language "und"
        mdhd.u16(0); // quality

        let mdia = Atom::with(
            b"mdia",
            &[
                mdhd.finish(),
                handler(b"mhlr", b"vide", "VideoHandler"),
                self.minf(&durations),
            ],
        );
        let trak = Atom::with(b"trak", &[tkhd.finish(), mdia]);
        Atom::with(b"moov", &[mvhd.finish(), trak])
    }

    fn minf(&self, durations: &[u32]) -> Vec<u8> {
        let mut vmhd = Atom::full(b"vmhd", 0, 0x1);
        vmhd.u16(0x40); // graphics mode: dither copy
        vmhd.zeros(6); // opcolor

        let mut dref = Atom::full(b"dref", 0, 0);
        dref.u32(1);
        dref.bytes(&Atom::full(b"alis", 0, 0x1).finish()); // media is in this file
        let dinf = Atom::with(b"dinf", &[dref.finish()]);

        let stbl = Atom::with(
            b"stbl",
            &[
                self.stsd(),
                stts(durations),
                stsc(),
                self.stsz(),
                self.co64(),
            ],
        );
        Atom::with(
            b"minf",
            &[
                vmhd.finish(),
                handler(b"dhlr", b"alis", "DataHandler"),
                dinf,
                stbl,
            ],
        )
    }

    fn stsd(&self) -> Vec<u8> {
        let mut jpeg = Atom::new(b"jpeg");
        jpeg.zeros(6);
        jpeg.u16(1); // data reference index
        jpeg.zeros(4); // version, revision
        jpeg.zeros(4); // vendor
        jpeg.u32(0); // temporal quality
        jpeg.u32(0x200); // spatial quality: codecNormalQuality
        jpeg.u16(self.width);
        jpeg.u16(self.height);
        jpeg.u32(0x0048_0000); // 72 dpi horizontal
        jpeg.u32(0x0048_0000); // 72 dpi vertical
        jpeg.u32(0); // data size
        jpeg.u16(1); // frames per sample
        let name = b"Photo - JPEG"; // Pascal string in a 32-byte field
        jpeg.u8(name.len() as u8);
        jpeg.bytes(name);
        jpeg.zeros(31 - name.len());
        jpeg.u16(24); // depth
        jpeg.u16(0xffff); // color table id: none

        let mut stsd = Atom::full(b"stsd", 0, 0);
        stsd.u32(1);
        stsd.bytes(&jpeg.finish());
        stsd.finish()
    }

    fn stsz(&self) -> Vec<u8> {
        let mut a = Atom::full(b"stsz", 0, 0);
        a.u32(0); // sizes vary, listed below
        a.u32(self.sizes.len() as u32);
        for &s in &self.sizes {
            a.u32(s);
        }
        a.finish()
    }

    fn co64(&self) -> Vec<u8> {
        let mut a = Atom::full(b"co64", 0, 0);
        a.u32(self.offsets.len() as u32);
        for &o in &self.offsets {
            a.u64(o);
        }
        a.finish()
    }
}

/// Units of the movie and track headers' overall duration.
const MOVIE_TIMESCALE: u32 = 1_000;

/// Units of per-frame durations; 90 kHz is the conventional video clock.
const MEDIA_TIMESCALE: u32 = 90_000;

/// Seconds from the QuickTime epoch (1904) to the Unix epoch (1970).
const QT_EPOCH_OFFSET: u64 = 2_082_844_800;

/// Time-to-sample table, run-length encoded.
fn stts(durations: &[u32]) -> Vec<u8> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        match runs.last_mut() {
            Some((n, last)) if *last == d => *n += 1,
            _ => runs.push((1, d)),
        }
    }
    let mut a = Atom::full(b"stts", 0, 0);
    a.u32(runs.len() as u32);
    for (n, d) in runs {
        a.u32(n);
        a.u32(d);
    }
    a.finish()
}

/// Sample-to-chunk table: every chunk holds exactly one frame.
fn stsc() -> Vec<u8> {
    let mut a = Atom::full(b"stsc", 0, 0);
    a.u32(1);
    a.u32(1); // first chunk
    a.u32(1); // samples per chunk
    a.u32(1); // sample description id
    a.finish()
}

fn handler(kind: &[u8; 4], subtype: &[u8; 4], name: &str) -> Vec<u8> {
    let mut a = Atom::full(b"hdlr", 0, 0);
    a.bytes(kind);
    a.bytes(subtype);
    a.zeros(12); // manufacturer, flags, flags mask
    a.u8(name.len() as u8); // Pascal string
    a.bytes(name.as_bytes());
    a.finish()
}

/// A big-endian atom under construction; its size is filled in by finish().
struct Atom(Vec<u8>);

impl Atom {
    fn new(kind: &[u8; 4]) -> Self {
        let mut v = vec![0; 4];
        v.extend_from_slice(kind);
        Self(v)
    }

    fn full(kind: &[u8; 4], version: u8, flags: u32) -> Self {
        let mut a = Self::new(kind);
        a.u32(u32::from(version) << 24 | flags);
        a
    }

    fn with(kind: &[u8; 4], children: &[Vec<u8>]) -> Vec<u8> {
        let mut a = Self::new(kind);
        for c in children {
            a.bytes(c);
        }
        a.finish()
    }

    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.bytes(&v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_be_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_be_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn zeros(&mut self, n: usize) {
        self.0.resize(self.0.len() + n, 0);
    }

    /// The identity transform, in QuickTime's 16.16 / 2.30 fixed point.
    fn matrix(&mut self) {
        for v in [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            self.u32(v);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let n = self.0.len() as u32;
        self.0[..4].copy_from_slice(&n.to_be_bytes());
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be32(b: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
    }
    fn be64(b: &[u8], at: usize) -> u64 {
        u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn atoms_tile_the_file_and_index_addresses_every_frame() {
        let path = std::env::temp_dir().join(format!("movtest-{}.mov", std::process::id()));
        let frames: [&[u8]; 3] = [b"first", b"second!", b"3"];
        let mut w = MovWriter::create(&path, 64, 48).unwrap();
        for (i, f) in frames.iter().enumerate() {
            w.write_frame(f, Duration::from_millis(33 * i as u64))
                .unwrap();
        }
        w.finish().unwrap();
        let data = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let mut at = 0;
        let mut kinds = Vec::new();
        while at < data.len() {
            let mut size = u64::from(be32(&data, at));
            if size == 1 {
                size = be64(&data, at + 8);
            }
            kinds.push(data[at + 4..at + 8].to_vec());
            at += size as usize;
        }
        assert_eq!(at, data.len());
        assert_eq!(
            kinds,
            [b"ftyp".to_vec(), b"mdat".to_vec(), b"moov".to_vec()]
        );

        let find = |tag: &[u8]| data.windows(4).position(|w| w == tag).unwrap();
        let (stsz, co64) = (find(b"stsz"), find(b"co64"));
        assert_eq!(be32(&data, stsz + 12), 3);
        assert_eq!(be32(&data, co64 + 8), 3);
        for (i, f) in frames.iter().enumerate() {
            let size = be32(&data, stsz + 16 + 4 * i) as usize;
            let off = be64(&data, co64 + 12 + 8 * i) as usize;
            assert_eq!(&data[off..off + size], *f);
        }
    }
}
