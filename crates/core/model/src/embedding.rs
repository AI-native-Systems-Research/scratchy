// SPDX-License-Identifier: Apache-2.0
//! Embedding utilities: pooling strategy detection and configuration.
//!
//! Used by embedding endpoints to configure how model hidden states are
//! converted into embedding vectors. The actual pooling/normalization is
//! performed by each backend (MLX, CUDA) using native tensor ops.

/// Pooling strategy for extracting embeddings from hidden states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the last token's hidden state (default for decoder models).
    Last,
    /// Use the first token's hidden state (CLS token for encoder models).
    Cls,
    /// Average all token hidden states.
    Mean,
    /// Return all token hidden states (ColBERT multi-vector).
    AllTokens,
}

impl std::str::FromStr for PoolingStrategy {
    type Err = String;

    /// Parse a pooling strategy from a string.
    ///
    /// Accepts: "last", "cls", "mean" (case-insensitive).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "last" => Ok(Self::Last),
            "cls" => Ok(Self::Cls),
            "mean" => Ok(Self::Mean),
            "all" | "all_tokens" => Ok(Self::AllTokens),
            other => Err(format!("unknown pooling strategy: {other}")),
        }
    }
}

/// Detect pooling strategy from a sentence-transformers `1_Pooling/config.json`.
///
/// The config has boolean fields like `pooling_mode_mean_tokens`,
/// `pooling_mode_cls_token`, `pooling_mode_lasttoken`. The first
/// `true` field wins.
///
/// Returns `None` if the file doesn't exist or has no recognized mode.
pub fn detect_pooling_strategy(model_dir: &std::path::Path) -> Option<PoolingStrategy> {
    let config_path = model_dir.join("1_Pooling").join("config.json");
    let data = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;

    if json
        .get("pooling_mode_mean_tokens")
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return Some(PoolingStrategy::Mean);
    }
    if json.get("pooling_mode_cls_token").and_then(|v| v.as_bool()) == Some(true) {
        return Some(PoolingStrategy::Cls);
    }
    if json.get("pooling_mode_lasttoken").and_then(|v| v.as_bool()) == Some(true) {
        return Some(PoolingStrategy::Last);
    }

    None
}

/// Resolve the `--pooling-strategy` CLI string against a model
/// directory: an explicit name parses, anything else (`"auto"`) is
/// detected from `1_Pooling/config.json` and falls back to `Last`.
///
/// The string is decoded HERE and nowhere else — every backend takes
/// its `PoolingStrategy` from this one boundary.
pub fn resolve_pooling_strategy(configured: &str, model_dir: &std::path::Path) -> PoolingStrategy {
    configured
        .parse()
        .unwrap_or_else(|_| detect_pooling_strategy(model_dir).unwrap_or(PoolingStrategy::Last))
}

/// Pool `[num_tokens, hidden]` row-major f32 hidden states into an
/// L2-normalized embedding under `strategy`.
///
/// Backend-neutral: a backend's only job is to land the rows in f32
/// host memory. `rows` must hold exactly `num_tokens * hidden` values.
pub fn pool_rows(
    rows: &[f32],
    num_tokens: usize,
    hidden: usize,
    strategy: PoolingStrategy,
) -> Vec<Vec<f32>> {
    assert_eq!(
        rows.len(),
        num_tokens * hidden,
        "pool_rows: {} values for {num_tokens}x{hidden}",
        rows.len()
    );
    let row = |i: usize| rows[i * hidden..(i + 1) * hidden].to_vec();
    let pooled = match strategy {
        PoolingStrategy::AllTokens => (0..num_tokens).map(row).collect(),
        PoolingStrategy::Cls => vec![row(0)],
        PoolingStrategy::Last => vec![row(num_tokens - 1)],
        PoolingStrategy::Mean => {
            let n = num_tokens as f32;
            let mut mean = vec![0.0f32; hidden];
            for i in 0..num_tokens {
                for (j, v) in mean.iter_mut().enumerate() {
                    *v += rows[i * hidden + j] / n;
                }
            }
            vec![mean]
        }
    };
    pooled.into_iter().map(l2_normalize).collect()
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_rows_selects_and_normalizes() {
        // 2 tokens x 3 dims.
        let rows = [3.0, 0.0, 4.0, 0.0, 6.0, 0.0];
        assert_eq!(
            pool_rows(&rows, 2, 3, PoolingStrategy::Cls),
            vec![vec![0.6, 0.0, 0.8]]
        );
        assert_eq!(
            pool_rows(&rows, 2, 3, PoolingStrategy::Last),
            vec![vec![0.0, 1.0, 0.0]]
        );
        assert_eq!(pool_rows(&rows, 2, 3, PoolingStrategy::AllTokens).len(), 2);
        let mean = pool_rows(&rows, 2, 3, PoolingStrategy::Mean).remove(0);
        // mean = [1.5, 3.0, 2.0], norm = sqrt(2.25 + 9 + 4).
        let n = (15.25f32).sqrt();
        for (got, want) in mean.iter().zip([1.5 / n, 3.0 / n, 2.0 / n]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn test_pooling_strategy_from_str() {
        assert_eq!(
            "last".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Last
        );
        assert_eq!(
            "Last".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Last
        );
        assert_eq!(
            "cls".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Cls
        );
        assert_eq!(
            "CLS".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Cls
        );
        assert_eq!(
            "mean".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Mean
        );
        assert_eq!(
            "MEAN".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Mean
        );
        assert_eq!(
            "all".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::AllTokens
        );
        assert_eq!(
            "all_tokens".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::AllTokens
        );
        assert_eq!(
            "ALL".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::AllTokens
        );
        assert!("unknown".parse::<PoolingStrategy>().is_err());
    }

    #[test]
    fn test_detect_pooling_strategy_mean() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": false}"#,
        )
        .unwrap();
        assert_eq!(
            detect_pooling_strategy(dir.path()),
            Some(PoolingStrategy::Mean)
        );
    }

    #[test]
    fn test_detect_pooling_strategy_cls() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_mean_tokens": false, "pooling_mode_cls_token": true}"#,
        )
        .unwrap();
        assert_eq!(
            detect_pooling_strategy(dir.path()),
            Some(PoolingStrategy::Cls)
        );
    }

    #[test]
    fn test_detect_pooling_strategy_last() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_lasttoken": true}"#,
        )
        .unwrap();
        assert_eq!(
            detect_pooling_strategy(dir.path()),
            Some(PoolingStrategy::Last)
        );
    }

    #[test]
    fn test_detect_pooling_strategy_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(detect_pooling_strategy(dir.path()), None);
    }

    #[test]
    fn test_detect_pooling_strategy_all_false() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_mean_tokens": false, "pooling_mode_cls_token": false}"#,
        )
        .unwrap();
        assert_eq!(detect_pooling_strategy(dir.path()), None);
    }
}
