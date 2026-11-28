// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Parquet → JSON rows, for the HuggingFace-hosted RAG accuracy datasets
//! (`hotpotqa`, `multihop`, `msmarco`). Each of those subcommands caches the
//! rows as JSON and parses its queries out of them, so the JSON shape this
//! produces for nested columns (structs of lists, lists of lists) is what their
//! parsers read.

use std::path::Path;

use anyhow::Result;

/// Read a parquet file and return rows as JSON values, each row straight from parquet's record
/// reader (`Row::to_json_value`).
pub(crate) fn parquet_to_json_records(parquet_path: &Path) -> Result<Vec<serde_json::Value>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    eprintln!("Reading parquet...");
    let reader = SerializedFileReader::new(std::fs::File::open(parquet_path)?)?;
    let records = (reader.get_row_iter(None)?)
        .map(|row| Ok(row?.to_json_value()))
        .collect::<Result<Vec<_>>>()?;
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
    use parquet::file::properties::WriterProperties;

    /// Write `rows` as a parquet file with `schema`, in row groups of at most `group_rows`.
    pub(crate) fn parquet_file_in_groups(
        schema: Schema,
        rows: &[serde_json::Value],
        group_rows: usize,
    ) -> tempfile::NamedTempFile {
        let schema = Arc::new(schema);
        let mut decoder = ReaderBuilder::new(schema.clone()).build_decoder().unwrap();
        decoder.serialize(rows).unwrap();
        let batch = decoder.flush().unwrap().unwrap();

        let file = tempfile::NamedTempFile::new().unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(group_rows))
            .build();
        let mut writer = ArrowWriter::try_new(file.reopen().unwrap(), schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        file
    }

    /// [`parquet_file_in_groups`], in one row group, read back through
    /// [`super::parquet_to_json_records`].
    pub(crate) fn through_parquet(
        schema: Schema,
        rows: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        let file = parquet_file_in_groups(schema, rows, rows.len().max(1));
        super::parquet_to_json_records(file.path()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use serde_json::json;

    use super::fixture::parquet_file_in_groups;
    use super::parquet_to_json_records;

    /// Every row group's rows must land in the one JSON array, in file order.
    #[test]
    fn rows_span_row_groups_in_order() {
        let rows: Vec<_> = (0..2_500).map(|n| json!({ "n": n })).collect();
        let schema = Schema::new(vec![Field::new("n", DataType::Int64, false)]);
        let file = parquet_file_in_groups(schema, &rows, 1_000);

        let groups = SerializedFileReader::new(file.reopen().unwrap())
            .unwrap()
            .metadata()
            .num_row_groups();
        assert!(groups > 1, "{} rows wrote {groups} row group", rows.len());

        assert_eq!(parquet_to_json_records(file.path()).unwrap(), rows);
    }
}
