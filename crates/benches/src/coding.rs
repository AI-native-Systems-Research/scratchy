// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The coding workloads `scr bench serve` sends ([`crate::workloads`]):
//!
//! - **Polyglot**, the Python exercises of Aider's polyglot benchmark (34 Exercism problems), sent
//!   as Aider sends them, and scored by running each answer against the exercise's own unit
//!   tests: pass@1, one attempt, no second round with the test errors.
//! - **SWE-bench Lite**, its oracle prompts (the issue and the files its fix touches), sent as
//!   written. Prompts only: a patch is scored in the repo's own environment, minutes a task.
//!
//! A Polyglot answer is model-written code, run on the machine benchmarking it. It runs under
//! `sandbox-exec` (no network, no file writes outside its own temporary directory), isolated from
//! the user's Python setup (`-I`), and stopped after [`TEST_TIMEOUT`].

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

/// Aider's polyglot benchmark, at the commit its exercises are read from.
const POLYGLOT_COMMIT: &str = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f";

/// Its Python exercises: the name, and whether it has `instructions.append.md`. `paasio`'s tests
/// also import `test_utils.py`.
const POLYGLOT_PYTHON: [(&str, bool); 34] = [
    ("affine-cipher", true),
    ("beer-song", false),
    ("book-store", false),
    ("bottle-song", false),
    ("bowling", true),
    ("connect", false),
    ("dominoes", false),
    ("dot-dsl", true),
    ("food-chain", false),
    ("forth", true),
    ("go-counting", true),
    ("grade-school", true),
    ("grep", false),
    ("hangman", true),
    ("list-ops", false),
    ("paasio", false),
    ("phone-number", true),
    ("pig-latin", false),
    ("poker", false),
    ("pov", true),
    ("proverb", true),
    ("react", false),
    ("rest-api", false),
    ("robot-name", false),
    ("scale-generator", false),
    ("sgf-parsing", true),
    ("simple-linked-list", true),
    ("transpose", false),
    ("tree-building", true),
    ("two-bucket", true),
    ("variable-length-quantity", true),
    ("wordy", true),
    ("zebra-puzzle", false),
    ("zipper", false),
];

/// The exercises with an `introduction.md`, which Aider sends before the instructions.
const POLYGLOT_INTRODUCED: [&str; 3] = ["pig-latin", "simple-linked-list", "zebra-puzzle"];

/// Aider's addendum to every exercise's instructions (`benchmark/prompts.py`), word for word.
const AIDER_ADDENDUM: &str = "\n####\n\nUse the above instructions to modify the supplied files: \
     {file_list}\nDon't change the names of existing functions or classes, as they may be \
     referenced from other code like unit tests, etc.\nOnly use standard libraries, don't suggest \
     installing any packages.\n";

/// How the answer comes back: the whole file, as Aider's `whole` edit format asks, so it can be
/// run as it is.
const WHOLE_FILE: &str = "Return the complete updated {file}, in one ```python fenced code block.";

/// The answer cap of a Polyglot exercise: a whole solution file, the longest of which (the
/// reference `book-store`, `zebra-puzzle`) run to a few hundred tokens.
pub(crate) const POLYGLOT_MAX_TOKENS: usize = 2048;

/// The answer cap of a SWE-bench patch. Its loader needs the `datasets` feature.
#[cfg(feature = "datasets")]
pub(crate) const SWE_BENCH_MAX_TOKENS: usize = 1024;

/// How long an answer's tests may run.
pub(crate) const TEST_TIMEOUT: Duration = Duration::from_secs(60);

/// What a Polyglot answer is run against: its exercise's solution file name, its tests, and any
/// other file the tests import.
#[derive(Clone, Debug)]
pub(crate) struct Exercise {
    pub(crate) name: String,
    pub(crate) solution: String,
    /// (file name, contents): the test file first.
    pub(crate) files: Vec<(String, String)>,
}

impl Exercise {
    /// The test module the runner loads: the test file's name without `.py`.
    fn test_module(&self) -> &str {
        self.files[0].0.trim_end_matches(".py")
    }
}

/// `~/.cache/scr-bench/<name>`.
fn cache_dir(name: &str) -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .context("Could not determine cache directory")?
        .join("scr-bench")
        .join(name);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// One file of the polyglot benchmark at [`POLYGLOT_COMMIT`], downloaded once.
fn polyglot_file(agent: &crate::http::Agent, cache: &Path, path: &str) -> Result<String> {
    let local = cache.join(path);
    if let Ok(text) = std::fs::read_to_string(&local) {
        return Ok(text);
    }
    let url = format!(
        "https://raw.githubusercontent.com/Aider-AI/polyglot-benchmark/{POLYGLOT_COMMIT}/{path}"
    );
    let text = crate::http::get_text(agent, &url).with_context(|| format!("fetching {url}"))?;
    std::fs::create_dir_all(local.parent().unwrap())?;
    std::fs::write(&local, &text)?;
    Ok(text)
}

/// The Python exercises of Aider's polyglot benchmark, as tasks: each prompt as Aider builds it
/// (the introduction, the instructions, the instructions' Python addition, then Aider's addendum),
/// with the stub file and a request for the whole file back; scored by its own tests.
pub(crate) fn polyglot_tasks() -> Result<Vec<crate::workloads::Task>> {
    let cache = cache_dir(&format!("polyglot/{POLYGLOT_COMMIT}"))?;
    let agent = crate::http::agent(false);
    let mut tasks = Vec::with_capacity(POLYGLOT_PYTHON.len());
    for (name, appended) in POLYGLOT_PYTHON {
        let dir = format!("python/exercises/practice/{name}");
        let get = |file: &str| polyglot_file(&agent, &cache, &format!("{dir}/{file}"));
        let module = name.replace('-', "_");
        let (solution, test) = (format!("{module}.py"), format!("{module}_test.py"));
        let mut instructions = String::new();
        if POLYGLOT_INTRODUCED.contains(&name) {
            instructions += &get(".docs/introduction.md")?;
        }
        instructions += &get(".docs/instructions.md")?;
        if appended {
            instructions += &get(".docs/instructions.append.md")?;
        }
        let mut files = vec![(test.clone(), get(&test)?)];
        if name == "paasio" {
            files.push(("test_utils.py".into(), get("test_utils.py")?));
        }
        let stub = get(&solution)?;
        let exercise = Exercise {
            name: name.to_string(),
            solution,
            files,
        };
        tasks.push(polyglot_task(&instructions, &stub, exercise));
    }
    Ok(tasks)
}

/// A Polyglot exercise as a task: its instructions with Aider's addendum, then the stub file.
pub(crate) fn polyglot_task(
    instructions: &str,
    stub: &str,
    exercise: Exercise,
) -> crate::workloads::Task {
    let file = &exercise.solution;
    let prompt = format!(
        "{instructions}{}\n{file}\n```python\n{stub}```\n\n{}",
        AIDER_ADDENDUM.replace("{file_list}", file),
        WHOLE_FILE.replace("{file}", file),
    );
    crate::workloads::Task {
        messages: vec![serde_json::json!({"role": "user", "content": prompt})],
        spans_input: None,
        max_tokens: POLYGLOT_MAX_TOKENS,
        score: crate::workloads::Score::Tests(Box::new(exercise)),
    }
}

/// SWE-bench Lite's oracle prompts, the issue and the files its fix touches, each as written:
/// prompts only, unscored. Downloaded once; the fields kept are the instance and its prompt.
#[cfg(feature = "datasets")]
pub(crate) fn swe_bench_tasks() -> Result<Vec<crate::workloads::Task>> {
    // The dataset's revision, so every run sends the same prompts.
    const REVISION: &str = "6fc5a85316392edbd1c2779120eeea362c535c82";
    let cache = cache_dir("swe-bench-lite-oracle")?;
    let json = cache.join(format!("test-{REVISION}.json"));
    let rows: Vec<(String, String)> = match std::fs::read_to_string(&json) {
        Ok(text) => serde_json::from_str(&text)?,
        Err(_) => {
            eprintln!("Downloading SWE-bench Lite (oracle)...");
            let url = format!(
                "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_oracle/resolve/\
                 {REVISION}/data/test-00000-of-00001.parquet"
            );
            let parquet = cache.join("test.parquet");
            let bytes = crate::http::get_bytes(&crate::http::agent(false), &url)?;
            std::fs::write(&parquet, &bytes)?;
            let records = crate::parquet_records::parquet_to_json_records(&parquet)?;
            let rows: Vec<(String, String)> = (records.iter())
                .filter_map(|r| {
                    Some((
                        r["instance_id"].as_str()?.into(),
                        r["text"].as_str()?.into(),
                    ))
                })
                .collect();
            std::fs::write(&json, serde_json::to_string(&rows)?)?;
            let _ = std::fs::remove_file(&parquet);
            rows
        }
    };
    Ok(rows
        .into_iter()
        .map(|(_, text)| swe_bench_task(text))
        .collect())
}

/// A SWE-bench prompt as a task: one user message, unscored.
#[cfg(feature = "datasets")]
pub(crate) fn swe_bench_task(text: String) -> crate::workloads::Task {
    crate::workloads::Task {
        messages: vec![serde_json::json!({"role": "user", "content": text})],
        spans_input: None,
        max_tokens: SWE_BENCH_MAX_TOKENS,
        score: crate::workloads::Score::None,
    }
}

/// The code an answer gives: its last fenced block, ```python or bare; the whole answer where it
/// has none, as a model may answer with the file alone.
pub(crate) fn answer_code(answer: &str) -> &str {
    let mut last = None;
    let mut rest = answer;
    let mut offset = 0;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let Some(nl) = after.find('\n') else { break };
        let body = &after[nl + 1..];
        let Some(close) = body.find("```") else { break };
        let start = offset + open + 3 + nl + 1;
        last = Some((start, start + close));
        let consumed = open + 3 + nl + 1 + close + 3;
        rest = &rest[consumed..];
        offset += consumed;
    }
    match last {
        Some((a, b)) => &answer[a..b],
        None => answer,
    }
}

/// What runs an answer's tests: a Python interpreter, under `sandbox-exec` where the machine has
/// it. None where neither is found, so a run without them records no accuracy rather than a
/// wrong one.
#[derive(Clone, Debug)]
pub(crate) struct TestRunner {
    python: PathBuf,
    sandbox: PathBuf,
}

/// The sandbox an answer's tests run in: no network, no file writes but in its own directory.
const SANDBOX_PROFILE: &str = "(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n\
     (allow file-write* (subpath (param \"WORK\")) (literal \"/dev/null\"))\n";

/// The test runner: loads the exercise's test module from the work directory and exits 0 only
/// when every test ran and passed.
const RUN_TESTS: &str = "import os, sys, unittest\nd = sys.argv[1]\nsys.path.insert(0, d)\n\
     os.chdir(d)\nsuite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[2])\n\
     r = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(suite)\n\
     sys.exit(0 if r.wasSuccessful() and r.testsRun > 0 else 1)\n";

impl TestRunner {
    /// `python` (a path, or a name on `PATH`) under `/usr/bin/sandbox-exec`; None where either is
    /// missing.
    pub(crate) fn find(python: &str) -> Option<Self> {
        let sandbox = PathBuf::from("/usr/bin/sandbox-exec");
        let python = which(python)?;
        (sandbox.exists()).then_some(Self { python, sandbox })
    }

    /// Whether `answer` passes `exercise`'s tests: its code written as the solution file beside
    /// the tests in a temporary directory, run there under the sandbox, stopped after
    /// [`TEST_TIMEOUT`]. A run that cannot start (the directory, the interpreter) is an error,
    /// not a failed answer.
    pub(crate) fn passes(&self, exercise: &Exercise, answer: &str) -> Result<bool> {
        self.passes_within(exercise, answer, TEST_TIMEOUT)
    }

    fn passes_within(&self, exercise: &Exercise, answer: &str, limit: Duration) -> Result<bool> {
        let work = WorkDir::new()?;
        let dir = work.0.clone();
        std::fs::write(dir.join(&exercise.solution), answer_code(answer))?;
        for (name, text) in &exercise.files {
            std::fs::write(dir.join(name), text)?;
        }
        std::fs::write(dir.join("_scr_run_tests.py"), RUN_TESTS)?;
        let profile = dir.join("_scr.sb");
        std::fs::write(&profile, SANDBOX_PROFILE)?;
        let mut child = std::process::Command::new(&self.sandbox)
            .arg("-f")
            .arg(&profile)
            .arg("-D")
            .arg(format!("WORK={}", dir.display()))
            .arg(&self.python)
            .args(["-I", "-B"])
            .arg(dir.join("_scr_run_tests.py"))
            .arg(&dir)
            .arg(exercise.test_module())
            .current_dir(&dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .with_context(|| format!("running {}'s tests", exercise.name))?;
        let start = std::time::Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                return Ok(status.success());
            }
            if start.elapsed() > limit {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// A test run's own directory under the system temp directory, removed when dropped: a path
/// this builds, never one from outside, so the removal cannot reach anything else.
struct WorkDir(PathBuf);

impl WorkDir {
    fn new() -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        // The sandbox matches real paths: on macOS /tmp is /private/tmp.
        let root = std::env::temp_dir().canonicalize()?;
        let name = format!(
            "scr-polyglot-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let dir = root.join(name);
        std::fs::create_dir(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `name` as a path: itself if it has a slash, else the first match on `PATH`.
fn which(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let p = PathBuf::from(name);
        return p.is_file().then_some(p);
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise() -> Exercise {
        Exercise {
            name: "adder".into(),
            solution: "adder.py".into(),
            files: vec![(
                "adder_test.py".into(),
                "import unittest\nfrom adder import add\n\n\
                 class T(unittest.TestCase):\n    def test_add(self):\n        \
                 self.assertEqual(add(2, 3), 5)\n"
                    .into(),
            )],
        }
    }

    #[test]
    fn the_code_is_the_answers_last_fenced_block() {
        let a = "Here:\n```python\nx = 1\n```\nThen better:\n```python\nx = 2\n```\nDone.";
        assert_eq!(answer_code(a), "x = 2\n");
        assert_eq!(answer_code("```\ny = 3\n```"), "y = 3\n", "a bare fence");
        assert_eq!(
            answer_code("z = 4\n"),
            "z = 4\n",
            "no fence: the whole answer"
        );
        assert_eq!(answer_code("```python\nunclosed"), "```python\nunclosed");
    }

    /// A SWE-bench prompt goes as written, one user message, and is not scored.
    #[cfg(feature = "datasets")]
    #[test]
    fn a_swe_bench_prompt_is_sent_as_written() {
        let t = swe_bench_task("<issue>x</issue>\n<code>y</code>\nI need you to solve".into());
        assert_eq!(t.messages.len(), 1);
        assert_eq!(
            t.messages[0]["content"],
            "<issue>x</issue>\n<code>y</code>\nI need you to solve"
        );
        assert_eq!(t.max_tokens, SWE_BENCH_MAX_TOKENS);
        assert!(!t.score.scored());
    }

    /// The prompt is Aider's: the instructions, its addendum naming the file, then the stub.
    #[test]
    fn a_polyglot_prompt_is_aiders() {
        let t = polyglot_task(
            "Add two numbers.\n",
            "def add(a, b):\n    pass\n",
            exercise(),
        );
        let p = t.messages[0]["content"].as_str().unwrap();
        assert!(p.starts_with("Add two numbers.\n\n####\n\nUse the above instructions"));
        assert!(p.contains("modify the supplied files: adder.py\n"));
        assert!(p.contains("Only use standard libraries, don't suggest installing any packages."));
        assert!(p.contains("adder.py\n```python\ndef add(a, b):\n    pass\n```"));
        assert!(p.ends_with(
            "Return the complete updated adder.py, in one ```python fenced code block."
        ));
        assert_eq!(t.messages.len(), 1);
        assert!(t.score.scored());
    }

    /// A right answer passes its tests, a wrong one and the stub fail, and code that tries the
    /// network or writes outside its directory gets nowhere. Skipped where the machine cannot run
    /// them (no Python, no sandbox-exec).
    #[test]
    fn an_answer_is_scored_by_its_tests_in_a_sandbox() {
        let Some(runner) = TestRunner::find("python3") else {
            eprintln!("skipped: no python3 or sandbox-exec");
            return;
        };
        let e = exercise();
        let right = "```python\ndef add(a, b):\n    return a + b\n```";
        assert!(runner.passes(&e, right).unwrap());
        assert!(
            !runner
                .passes(&e, "```python\ndef add(a, b):\n    return a - b\n```")
                .unwrap()
        );
        assert!(
            !runner
                .passes(&e, "```python\ndef add(a, b):\n    pass\n```")
                .unwrap()
        );
        assert!(!runner.passes(&e, "no code at all").unwrap());
        let home = dirs::home_dir().unwrap().join(".scr-sandbox-escape-test");
        let escape = format!(
            "```python\nimport socket\nopen({:?}, 'w').write('x')\n\
             socket.create_connection(('1.1.1.1', 80), timeout=3)\n\
             def add(a, b):\n    return a + b\n```",
            home.display().to_string()
        );
        assert!(
            !runner.passes(&e, &escape).unwrap(),
            "the write or the socket raised"
        );
        assert!(!home.exists(), "nothing written outside the work directory");
        let hang = "```python\nwhile True:\n    pass\n```";
        let start = std::time::Instant::now();
        let limit = Duration::from_secs(2);
        assert!(
            !runner.passes_within(&e, hang, limit).unwrap(),
            "stopped, and failed"
        );
        assert!(start.elapsed() < limit + Duration::from_secs(3));
    }
}
