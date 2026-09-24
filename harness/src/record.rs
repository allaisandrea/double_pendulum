//! The frames table a collection run writes, as an Arrow IPC stream.
//!
//! The stream format has no footer, so a file cut short by a crash is still
//! readable up to its last complete batch. Batches are written about once a
//! second, which bounds what a crash can lose.
//!
//! All times are `Duration(ns)` relative to the recording's t0.

use anyhow::{Context, Result};
use arrow_array::builder::{
    BooleanBuilder, DurationNanosecondBuilder, FixedSizeListBuilder, Float32Builder, Int8Builder,
    UInt64Builder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Position then orientation: x y z qw qx qy qz.
const POSE_LEN: i32 = 7;

/// One tag as seen in one frame.
#[derive(Clone, Debug)]
pub struct TagRow {
    /// x y z in metres, then the quaternion w x y z with w >= 0.
    pub pose: [f32; 7],
    pub err: f32,
    pub alt_err: Option<f32>,
    pub margin: f32,
}

/// One frame that went through detection.
#[derive(Clone, Debug)]
pub struct FrameRow {
    pub frame: u64,
    pub t_capture: i64,
    /// When the frame reached this program.
    pub t_arrival: i64,
    pub t_detect_start: i64,
    pub t_detected: i64,
    /// One entry per recorded tag id, in the order of the schema's columns.
    pub tags: Vec<Option<TagRow>>,
    /// Whether the policy acted on this frame. A frame that arrived while
    /// the policy was busy with an earlier one is skipped: its policy times
    /// are null and `action` is the one carried over.
    pub acted: bool,
    pub t_policy_start: Option<i64>,
    pub t_policy_done: Option<i64>,
    /// When the action was written to the serial port.
    pub t_sent: Option<i64>,
    /// The action in effect after this frame.
    pub action: i8,
}

/// Columns before and after the per-tag groups.
const HEAD: [&str; 5] = ["frame", "t_capture", "t_arrival", "t_detect_start", "t_detected"];
const TAIL: [&str; 5] = ["acted", "t_policy_start", "t_policy_done", "t_sent", "action"];
const PER_TAG: usize = 4;

fn duration() -> DataType {
    DataType::Duration(TimeUnit::Nanosecond)
}

fn pose_type() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), POSE_LEN)
}

fn frames_schema(tag_ids: &[usize], metadata: HashMap<String, String>) -> SchemaRef {
    let mut fields = vec![Field::new(HEAD[0], DataType::UInt64, false)];
    fields.extend(HEAD[1..].iter().map(|n| Field::new(*n, duration(), false)));
    for id in tag_ids {
        fields.push(Field::new(format!("tag{id}_pose"), pose_type(), true));
        fields.push(Field::new(format!("tag{id}_err"), DataType::Float32, true));
        fields.push(Field::new(format!("tag{id}_alt_err"), DataType::Float32, true));
        fields.push(Field::new(format!("tag{id}_margin"), DataType::Float32, true));
    }
    fields.push(Field::new(TAIL[0], DataType::Boolean, false));
    fields.extend(TAIL[1..4].iter().map(|n| Field::new(*n, duration(), true)));
    fields.push(Field::new(TAIL[4], DataType::Int8, false));
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

fn pose_builder() -> FixedSizeListBuilder<Float32Builder> {
    FixedSizeListBuilder::new(Float32Builder::new(), POSE_LEN)
        .with_field(Field::new("item", DataType::Float32, false))
}

fn frames_batch(schema: &SchemaRef, rows: &[FrameRow]) -> Result<RecordBatch> {
    let n_tags = (schema.fields().len() - HEAD.len() - TAIL.len()) / PER_TAG;
    let mut frame = UInt64Builder::new();
    let mut times: Vec<_> = (1..HEAD.len()).map(|_| DurationNanosecondBuilder::new()).collect();
    let mut poses: Vec<_> = (0..n_tags).map(|_| pose_builder()).collect();
    let mut errs: Vec<_> = (0..n_tags).map(|_| Float32Builder::new()).collect();
    let mut alts: Vec<_> = (0..n_tags).map(|_| Float32Builder::new()).collect();
    let mut margins: Vec<_> = (0..n_tags).map(|_| Float32Builder::new()).collect();
    let mut acted = BooleanBuilder::new();
    let mut policy_times: Vec<_> = (0..3).map(|_| DurationNanosecondBuilder::new()).collect();
    let mut action = Int8Builder::new();

    for r in rows {
        anyhow::ensure!(r.tags.len() == n_tags, "row has {} tags, schema {n_tags}", r.tags.len());
        frame.append_value(r.frame);
        for (b, t) in times.iter_mut().zip([r.t_capture, r.t_arrival, r.t_detect_start, r.t_detected]) {
            b.append_value(t);
        }
        for (i, tag) in r.tags.iter().enumerate() {
            match tag {
                Some(t) => {
                    poses[i].values().append_slice(&t.pose);
                    poses[i].append(true);
                    errs[i].append_value(t.err);
                    alts[i].append_option(t.alt_err);
                    margins[i].append_value(t.margin);
                }
                None => {
                    // A null fixed-size list still takes up its slots.
                    poses[i].values().append_nulls(POSE_LEN as usize);
                    poses[i].append(false);
                    errs[i].append_null();
                    alts[i].append_null();
                    margins[i].append_null();
                }
            }
        }
        acted.append_value(r.acted);
        for (b, t) in policy_times.iter_mut().zip([r.t_policy_start, r.t_policy_done, r.t_sent]) {
            b.append_option(t);
        }
        action.append_value(r.action);
    }

    let mut columns: Vec<ArrayRef> = vec![Arc::new(frame.finish())];
    columns.extend(times.iter_mut().map(|b| Arc::new(b.finish()) as ArrayRef));
    for i in 0..n_tags {
        columns.push(Arc::new(poses[i].finish()));
        columns.push(Arc::new(errs[i].finish()));
        columns.push(Arc::new(alts[i].finish()));
        columns.push(Arc::new(margins[i].finish()));
    }
    columns.push(Arc::new(acted.finish()));
    columns.extend(policy_times.iter_mut().map(|b| Arc::new(b.finish()) as ArrayRef));
    columns.push(Arc::new(action.finish()));
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

/// Buffers frame rows and writes them to an IPC stream in batches.
pub struct Table {
    schema: SchemaRef,
    writer: StreamWriter<BufWriter<File>>,
    rows: Vec<FrameRow>,
    written: u64,
}

impl Table {
    /// Creates the frames table at `path`, with one column group per tag id
    /// and `metadata` in its schema.
    pub fn create(path: &Path, tag_ids: &[usize], metadata: HashMap<String, String>) -> Result<Self> {
        let schema = frames_schema(tag_ids, metadata);
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let writer = StreamWriter::try_new(BufWriter::new(file), &schema)?;
        Ok(Self {
            schema,
            writer,
            rows: Vec::new(),
            written: 0,
        })
    }

    pub fn push(&mut self, row: FrameRow) {
        self.rows.push(row);
    }

    /// Writes the buffered rows as one batch and flushes it to disk.
    pub fn flush(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let batch = frames_batch(&self.schema, &self.rows)?;
        self.writer.write(&batch)?;
        self.writer.flush()?;
        self.written += self.rows.len() as u64;
        self.rows.clear();
        Ok(())
    }

    /// Flushes and writes the end-of-stream marker. Returns rows written.
    pub fn finish(mut self) -> Result<u64> {
        self.flush()?;
        self.writer.finish()?;
        Ok(self.written)
    }
}

/// Signed nanoseconds from `t0` to `t`, both on the monotonic clock.
pub fn rel_ns(t: Duration, t0: Duration) -> i64 {
    t.as_nanos() as i64 - t0.as_nanos() as i64
}

/// The detector's pose as the 7 recorded floats: position, then a unit
/// quaternion with w >= 0. q and -q are the same rotation; picking one
/// stops the stored values flipping sign between frames for no reason.
pub fn pose_row(t: [f64; 3], q: [f64; 4]) -> [f32; 7] {
    let s = if q[0] < 0.0 { -1.0 } else { 1.0 };
    [
        t[0] as f32,
        t[1] as f32,
        t[2] as f32,
        (s * q[0]) as f32,
        (s * q[1]) as f32,
        (s * q[2]) as f32,
        (s * q[3]) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{DurationNanosecondType, Float32Type, Int8Type};
    use arrow_array::Array;
    use arrow_ipc::reader::StreamReader;

    #[test]
    fn quaternions_are_stored_with_w_non_negative() {
        let r = pose_row([1.0, 2.0, 3.0], [-0.5, 0.5, -0.5, 0.5]);
        assert_eq!(r, [1.0, 2.0, 3.0, 0.5, -0.5, 0.5, -0.5]);
        let r = pose_row([0.0; 3], [0.5, 0.5, 0.5, 0.5]);
        assert_eq!(&r[3..], [0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn the_frames_table_survives_a_round_trip() {
        let dir = std::env::temp_dir().join(format!("record-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let meta = HashMap::from([("seed".to_string(), "1".to_string())]);

        let path = dir.join("frames.arrows");
        let mut table = Table::create(&path, &[0, 2], meta).unwrap();
        let seen = TagRow {
            pose: [0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
            err: 1e-6,
            alt_err: None,
            margin: 80.0,
        };
        table.push(FrameRow {
            frame: 7,
            t_capture: -5,
            t_arrival: 10,
            t_detect_start: 11,
            t_detected: 12,
            tags: vec![Some(seen.clone()), None],
            acted: true,
            t_policy_start: Some(13),
            t_policy_done: Some(14),
            t_sent: Some(15),
            action: -30,
        });
        table.flush().unwrap(); // two batches, to prove the stream concatenates
        table.push(FrameRow {
            frame: 8,
            t_capture: 30,
            t_arrival: 31,
            t_detect_start: 32,
            t_detected: 40,
            tags: vec![None, Some(seen)],
            acted: false,
            t_policy_start: None,
            t_policy_done: None,
            t_sent: None,
            action: -30,
        });
        assert_eq!(table.finish().unwrap(), 2);

        let reader = StreamReader::try_new(File::open(&path).unwrap(), None).unwrap();
        assert_eq!(reader.schema().metadata()["seed"], "1");
        let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 2);
        let (b, c) = (&batches[0], &batches[1]);
        let t = |b: &RecordBatch, n: &str| b.column_by_name(n).unwrap().as_primitive::<DurationNanosecondType>().value(0);
        assert_eq!(t(b, "t_capture"), -5);
        assert_eq!(t(b, "t_arrival"), 10);
        assert_eq!(t(c, "t_detected"), 40);
        assert_eq!(t(b, "t_sent"), 15);
        let pose = b.column_by_name("tag0_pose").unwrap().as_fixed_size_list();
        assert_eq!(pose.value(0).as_primitive::<Float32Type>().value(2), 0.9);
        assert!(b.column_by_name("tag2_pose").unwrap().is_null(0));
        assert!(b.column_by_name("tag0_alt_err").unwrap().is_null(0));
        assert!(c.column_by_name("tag0_pose").unwrap().is_null(0));
        assert!(b.column_by_name("acted").unwrap().as_boolean().value(0));
        assert!(!c.column_by_name("acted").unwrap().as_boolean().value(0));
        assert!(c.column_by_name("t_policy_start").unwrap().is_null(0));
        assert_eq!(c.column_by_name("action").unwrap().as_primitive::<Int8Type>().value(0), -30);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
