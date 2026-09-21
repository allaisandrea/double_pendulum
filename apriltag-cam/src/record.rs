//! The two tables a collection run writes, as Arrow IPC streams.
//!
//! The stream format has no footer, so a file cut short by a crash is still
//! readable up to its last complete batch. Batches are written about once a
//! second, which bounds what a crash can lose.
//!
//! All times are `Duration(ns)` relative to the grid's t0. A slot's time is
//! `slot * period`, so the actions table stores slots, not times.

use anyhow::{Context, Result};
use arrow_array::builder::{
    DurationNanosecondBuilder, FixedSizeListBuilder, Float32Builder, Int64Builder, Int8Builder,
    ListBuilder, UInt64Builder, UInt8Builder,
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
pub const POSE_LEN: i32 = 7;

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
pub struct ObsRow {
    pub frame: u64,
    pub t_capture: i64,
    pub t_detected: i64,
    /// One entry per recorded tag id, in the order of the schema's columns.
    pub tags: Vec<Option<TagRow>>,
    pub t_policy: Option<i64>,
    pub t_plan: Option<i64>,
    pub k_start: Option<i64>,
    pub plan: Option<Vec<i8>>,
}

/// One grid slot.
#[derive(Clone, Copy, Debug)]
pub struct ActRow {
    pub slot: i64,
    pub action: i8,
    /// The frame whose plan supplied the action; None for a gap.
    pub frame: Option<u64>,
    pub index: Option<u8>,
}

fn duration() -> DataType {
    DataType::Duration(TimeUnit::Nanosecond)
}

fn pose_type() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), POSE_LEN)
}

fn plan_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Int8, false)))
}

pub fn observations_schema(tag_ids: &[usize], metadata: HashMap<String, String>) -> SchemaRef {
    let mut fields = vec![
        Field::new("frame", DataType::UInt64, false),
        Field::new("t_capture", duration(), false),
        Field::new("t_detected", duration(), false),
    ];
    for id in tag_ids {
        fields.push(Field::new(format!("tag{id}_pose"), pose_type(), true));
        fields.push(Field::new(format!("tag{id}_err"), DataType::Float32, true));
        fields.push(Field::new(format!("tag{id}_alt_err"), DataType::Float32, true));
        fields.push(Field::new(format!("tag{id}_margin"), DataType::Float32, true));
    }
    fields.extend([
        Field::new("t_policy", duration(), true),
        Field::new("t_plan", duration(), true),
        Field::new("k_start", DataType::Int64, true),
        Field::new("plan", plan_type(), true),
    ]);
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

pub fn actions_schema(metadata: HashMap<String, String>) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("slot", DataType::Int64, false),
            Field::new("action", DataType::Int8, false),
            Field::new("frame", DataType::UInt64, true),
            Field::new("index", DataType::UInt8, true),
        ],
        metadata,
    ))
}

fn pose_builder() -> FixedSizeListBuilder<Float32Builder> {
    FixedSizeListBuilder::new(Float32Builder::new(), POSE_LEN)
        .with_field(Field::new("item", DataType::Float32, false))
}

fn plan_builder() -> ListBuilder<Int8Builder> {
    ListBuilder::new(Int8Builder::new()).with_field(Field::new("item", DataType::Int8, false))
}

pub fn observations_batch(schema: &SchemaRef, rows: &[ObsRow]) -> Result<RecordBatch> {
    let n_tags = (schema.fields().len() - 7) / 4;
    let mut frame = UInt64Builder::new();
    let mut t_capture = DurationNanosecondBuilder::new();
    let mut t_detected = DurationNanosecondBuilder::new();
    let mut poses: Vec<_> = (0..n_tags).map(|_| pose_builder()).collect();
    let mut errs: Vec<_> = (0..n_tags).map(|_| Float32Builder::new()).collect();
    let mut alts: Vec<_> = (0..n_tags).map(|_| Float32Builder::new()).collect();
    let mut margins: Vec<_> = (0..n_tags).map(|_| Float32Builder::new()).collect();
    let mut t_policy = DurationNanosecondBuilder::new();
    let mut t_plan = DurationNanosecondBuilder::new();
    let mut k_start = Int64Builder::new();
    let mut plan = plan_builder();

    for r in rows {
        anyhow::ensure!(r.tags.len() == n_tags, "row has {} tags, schema {n_tags}", r.tags.len());
        frame.append_value(r.frame);
        t_capture.append_value(r.t_capture);
        t_detected.append_value(r.t_detected);
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
        t_policy.append_option(r.t_policy);
        t_plan.append_option(r.t_plan);
        k_start.append_option(r.k_start);
        match &r.plan {
            Some(p) => {
                plan.values().append_slice(p);
                plan.append(true);
            }
            None => plan.append(false),
        }
    }

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(frame.finish()),
        Arc::new(t_capture.finish()),
        Arc::new(t_detected.finish()),
    ];
    for i in 0..n_tags {
        columns.push(Arc::new(poses[i].finish()));
        columns.push(Arc::new(errs[i].finish()));
        columns.push(Arc::new(alts[i].finish()));
        columns.push(Arc::new(margins[i].finish()));
    }
    columns.extend([
        Arc::new(t_policy.finish()) as ArrayRef,
        Arc::new(t_plan.finish()),
        Arc::new(k_start.finish()),
        Arc::new(plan.finish()),
    ]);
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

pub fn actions_batch(schema: &SchemaRef, rows: &[ActRow]) -> Result<RecordBatch> {
    let mut slot = Int64Builder::new();
    let mut action = Int8Builder::new();
    let mut frame = UInt64Builder::new();
    let mut index = UInt8Builder::new();
    for r in rows {
        slot.append_value(r.slot);
        action.append_value(r.action);
        frame.append_option(r.frame);
        index.append_option(r.index);
    }
    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(slot.finish()),
            Arc::new(action.finish()),
            Arc::new(frame.finish()),
            Arc::new(index.finish()),
        ],
    )?)
}

/// Buffers rows and writes them to an IPC stream in batches.
pub struct Table<R> {
    schema: SchemaRef,
    writer: StreamWriter<BufWriter<File>>,
    rows: Vec<R>,
    batch: fn(&SchemaRef, &[R]) -> Result<RecordBatch>,
    written: u64,
}

impl<R> Table<R> {
    pub fn create(
        path: &Path,
        schema: SchemaRef,
        batch: fn(&SchemaRef, &[R]) -> Result<RecordBatch>,
    ) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let writer = StreamWriter::try_new(BufWriter::new(file), &schema)?;
        Ok(Self {
            schema,
            writer,
            rows: Vec::new(),
            batch,
            written: 0,
        })
    }

    pub fn push(&mut self, row: R) {
        self.rows.push(row);
    }

    /// Writes the buffered rows as one batch and flushes it to disk.
    pub fn flush(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let batch = (self.batch)(&self.schema, &self.rows)?;
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
    fn both_tables_survive_a_round_trip() {
        let dir = std::env::temp_dir().join(format!("record-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let meta = HashMap::from([("period_ns".to_string(), "20000000".to_string())]);

        let obs_path = dir.join("observations.arrows");
        let mut obs = Table::create(
            &obs_path,
            observations_schema(&[0, 2], meta.clone()),
            observations_batch,
        )
        .unwrap();
        let seen = TagRow {
            pose: [0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
            err: 1e-6,
            alt_err: None,
            margin: 80.0,
        };
        obs.push(ObsRow {
            frame: 7,
            t_capture: -5,
            t_detected: 12,
            tags: vec![Some(seen.clone()), None],
            t_policy: Some(13),
            t_plan: Some(14),
            k_start: Some(3),
            plan: Some(vec![1, -2, 3]),
        });
        obs.flush().unwrap(); // two batches, to prove the stream concatenates
        obs.push(ObsRow {
            frame: 8,
            t_capture: 30,
            t_detected: 40,
            tags: vec![None, Some(seen)],
            t_policy: None,
            t_plan: None,
            k_start: None,
            plan: None,
        });
        assert_eq!(obs.finish().unwrap(), 2);

        let act_path = dir.join("actions.arrows");
        let mut act = Table::create(&act_path, actions_schema(meta), actions_batch).unwrap();
        act.push(ActRow { slot: 0, action: 0, frame: None, index: None });
        act.push(ActRow { slot: 1, action: -30, frame: Some(7), index: Some(4) });
        assert_eq!(act.finish().unwrap(), 2);

        let reader = StreamReader::try_new(File::open(&obs_path).unwrap(), None).unwrap();
        assert_eq!(reader.schema().metadata()["period_ns"], "20000000");
        let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 2);
        let b = &batches[0];
        assert_eq!(b.column_by_name("t_capture").unwrap().as_primitive::<DurationNanosecondType>().value(0), -5);
        let pose = b.column_by_name("tag0_pose").unwrap().as_fixed_size_list();
        assert_eq!(pose.value(0).as_primitive::<Float32Type>().value(2), 0.9);
        assert!(b.column_by_name("tag2_pose").unwrap().is_null(0));
        assert!(b.column_by_name("tag0_alt_err").unwrap().is_null(0));
        let plan = b.column_by_name("plan").unwrap().as_list::<i32>();
        assert_eq!(plan.value(0).as_primitive::<Int8Type>().values(), &[1, -2, 3]);
        assert!(batches[1].column_by_name("plan").unwrap().is_null(0));
        assert!(batches[1].column_by_name("tag0_pose").unwrap().is_null(0));

        let reader = StreamReader::try_new(File::open(&act_path).unwrap(), None).unwrap();
        let b = reader.map(|b| b.unwrap()).next().unwrap();
        assert!(b.column_by_name("frame").unwrap().is_null(0));
        assert_eq!(b.column_by_name("action").unwrap().as_primitive::<Int8Type>().value(1), -30);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
