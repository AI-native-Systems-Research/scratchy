//! The data files' one format: JSON, gzipped, named `<stem>.json.gz`.
//!
//! Every page that reads `data/` lists its folder and reads each file here, so
//! the format and its name are stated once.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The ending of every data file's name.
pub const EXT: &str = ".json.gz";

/// A data file's name, from its stem: `<stem>.json.gz`.
pub fn named(stem: &str) -> String {
    format!("{stem}{EXT}")
}

/// The data files directly in `dir`, sorted: each `*.json.gz` as its path, and
/// each file that looks like data but is not one (a plain `*.json`, a `*.gz`
/// that is not `*.json.gz`) as an error naming it, so a forgotten gzip fails
/// the build. Other entries (folders, editor files) are skipped. Fails only
/// when `dir` cannot be read.
pub fn list(dir: &Path) -> std::io::Result<Vec<Result<PathBuf, String>>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    Ok(paths
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().into_owned();
            if name.ends_with(EXT) {
                Some(Ok(p))
            } else if name.ends_with(".json") || name.ends_with(".gz") {
                Some(Err(format!(
                    "{name}: data files are gzipped JSON; gzip -9 -n it to <name>{EXT}"
                )))
            } else {
                None
            }
        })
        .collect())
}

/// The JSON text of one gzipped data file.
pub fn read_json_gz(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(file)
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    #[test]
    fn a_gzipped_file_reads_back_as_its_json() {
        let dir = std::env::temp_dir().join(format!("datafile-{}-read", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        gz.write_all(br#"{"a": 1}"#).unwrap();
        let path = dir.join(super::named("x"));
        std::fs::write(&path, gz.finish().unwrap()).unwrap();
        assert_eq!(super::read_json_gz(&path).unwrap(), r#"{"a": 1}"#);
        let listed = super::list(&dir).unwrap();
        assert_eq!(listed, vec![Ok(path)]);
    }
}
