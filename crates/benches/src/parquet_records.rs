// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Parquet → JSON rows, for the HuggingFace-hosted RAG accuracy datasets
//! (`hotpotqa`, `multihop`, `msmarco`). Each of those subcommands caches the
//! rows as JSON and parses its queries out of them, so the JSON shape this
//! produces for nested columns (structs of lists, lists of lists) is what their
//! parsers read.

use std::path::Path;

use anyhow::Result;

/// Read a parquet file and return rows as JSON values.
pub(crate) fn parquet_to_json_records(parquet_path: &Path) -> Result<Vec<serde_json::Value>> {
    use arrow::json::writer::{JsonArray, Writer};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    eprintln!("Reading parquet...");
    let file = std::fs::File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let batches: Vec<_> = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    let batch_refs: Vec<&_> = batches.iter().collect();

    let mut buf = Vec::new();
    let mut writer = Writer::<_, JsonArray>::new(&mut buf);
    writer.write_batches(&batch_refs)?;
    writer.finish()?;
    drop(writer);

    let records: Vec<serde_json::Value> = serde_json::from_slice(&buf)?;
    eprintln!("Read {} records.", records.len());
    Ok(records)
}

/// Test fixtures: rows written to a real parquet file under a declared schema.
#[cfg(test)]
pub(crate) mod fixture {
    use std::sync::Arc;

    use arrow::datatypes::Schema;
    use arrow::json::reader::ReaderBuilder;
    use parquet::arrow::ArrowWriter;

    /// Write `rows` as a parquet file with `schema`.
    pub(crate) fn parquet_file(
        schema: Schema,
        rows: &[serde_json::Value],
    ) -> tempfile::NamedTempFile {
        let schema = Arc::new(schema);
        let mut decoder = ReaderBuilder::new(schema.clone()).build_decoder().unwrap();
        decoder.serialize(rows).unwrap();
        let batch = decoder.flush().unwrap().unwrap();

        let file = tempfile::NamedTempFile::new().unwrap();
        let mut writer = ArrowWriter::try_new(file.reopen().unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        file
    }

    /// [`parquet_file`], read back through [`super::parquet_to_json_records`].
    pub(crate) fn through_parquet(
        schema: Schema,
        rows: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        let file = parquet_file(schema, rows);
        super::parquet_to_json_records(file.path()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use serde_json::json;

    use super::fixture::parquet_file;
    use super::parquet_to_json_records;

    /// Every batch the reader hands back must land in the one JSON array, in
    /// file order.
    #[test]
    fn rows_span_reader_batches_in_order() {
        let rows: Vec<_> = (0..2_500).map(|n| json!({ "n": n })).collect();
        let schema = Schema::new(vec![Field::new("n", DataType::Int64, false)]);
        let file = parquet_file(schema, &rows);

        let batches = ParquetRecordBatchReaderBuilder::try_new(file.reopen().unwrap())
            .unwrap()
            .build()
            .unwrap()
            .count();
        assert!(
            batches > 1,
            "{} rows came back in {batches} batch",
            rows.len()
        );

        assert_eq!(parquet_to_json_records(file.path()).unwrap(), rows);
    }
}
