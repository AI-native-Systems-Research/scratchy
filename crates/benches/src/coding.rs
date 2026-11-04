// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The coding workloads `scr bench serve` sends ([`crate::workloads`]):
//!
//! - **Polyglot**, the Python exercises of Aider's polyglot benchmark (34 Exercism problems), with
//!   Aider's instructions and addendum word for word, and scored by running each answer against
//!   the exercise's own unit tests: pass@1, one attempt, no second round with the test errors.
//!   Aider-derived, not Aider's own run: no system prompt, this crate's own request for the whole
//!   file, and the answer's last fenced block taken as its code, where Aider sends its `whole`
//!   format's prompts and takes the block named for the file. Its scores are not comparable with
//!   Aider's published ones.
//! - **SWE-bench Lite**, its oracle prompts (the issue and the files its fix touches), sent as
//!   written. Prompts only: a patch is scored in the repo's own environment, minutes a task.
//!
//! A Polyglot answer is model-written code, run on the machine benchmarking it. It runs under
//! `sandbox-exec` under a deny-default profile ([`SANDBOX_PROFILE`]: no network, no exec but the
//! interpreter, no fork, no reads or writes outside its own temporary directory and Python's own
//! files), isolated from the user's Python setup (`-I`), and stopped after [`TEST_TIMEOUT`] or
//! past [`TEST_MEMORY`]. It passes only on the tests' own report ([`RUN_TESTS`]), never on how the
//! run exited.

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

/// How the answer comes back: the whole file, so it can be run as it is. This crate's own
/// sentence, not Aider's `whole` format prompts.
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

/// The Python exercises of Aider's polyglot benchmark, as tasks: each prompt's text in Aider's
/// order (the introduction, the instructions, the instructions' Python addition, then Aider's
/// addendum), with the stub file and [`WHOLE_FILE`]'s request for the whole file back; scored by
/// its own tests. No system prompt, so its scores are Aider-derived, not Aider's.
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
    /// The interpreter binary itself, as it runs: a shim or framework launcher (`/usr/bin/python3`,
    /// `bin/python3.x`) resolved to the binary they end in, so the sandbox lets it alone exec.
    python: PathBuf,
    /// Its prefix (`sys.base_prefix`): the standard library and framework it reads.
    home: PathBuf,
    sandbox: PathBuf,
}

/// The sandbox an answer's tests run in: everything denied but what an interpreter running
/// `unittest` needs. It reads, or even stats, only its own directory, the interpreter's files and
/// the system's (`system.sb`: dyld, the shared cache, the system libraries); it writes only its
/// own directory; it execs only the interpreter, so it reaches no other binary, and with no Mach
/// lookups beyond `system.sb`'s, no service that would write for it (cfprefsd, launchd, the
/// pasteboard).
const SANDBOX_PROFILE: &str = "(version 1)\n(deny default)\n(import \"system.sb\")\n\
     (allow process-exec (literal (param \"PYTHON\")))\n\
     (allow file-read* (subpath (param \"WORK\")) (subpath (param \"PYTHON_HOME\")))\n\
     (allow file-write* (subpath (param \"WORK\")) (literal \"/dev/null\"))\n\
     (allow sysctl-read)\n";

/// The memory an answer's tests may take, every process of its group together: far more than any
/// exercise needs, far less than the server the matrix is timing beside it. Watched, not set as
/// an rlimit: macOS 26 refuses an RLIMIT_DATA or RLIMIT_AS below what a process has reserved at
/// start, which is far above this.
const TEST_MEMORY: u64 = 2 << 30;

/// Prints the interpreter binary as it runs (a shim or launcher resolved through), then its
/// prefix, both as real paths: the sandbox matches real paths, and Homebrew's prefix is a symlink
/// (`opt/python@3.x` to `Cellar/...`). Exits 0 only if `unittest` imports.
const PROBE: &str = "import ctypes, os, sys, unittest\n\
     b = ctypes.create_string_buffer(4096)\nn = ctypes.c_uint32(4096)\n\
     ctypes.CDLL(None)._NSGetExecutablePath(b, ctypes.byref(n))\n\
     print(os.path.realpath(b.value.decode()))\nprint(os.path.realpath(sys.base_prefix))\n";

/// The test runner, given the work directory, the test module and where to report. It reads a
/// one-run token on stdin, loads the exercise's test module (which imports the answer), puts back
/// any of `unittest`'s classes the answer changed as it loaded, runs the tests, and only then
/// writes its report ([`RESULT_FILE`]): the token, then the
/// tests run, failures, errors, skips, expected failures and unexpected successes. An answer that
/// ends the run as it loads (`sys.exit(0)`, `os._exit(0)`, `raise SystemExit`) or raises from
/// anywhere writes no report, so it fails, whatever its exit code; the runner then leaves at once
/// (`os._exit`), so no exit handler the answer set runs after the report. This stops an answer passing by
/// accident, or by an obvious patch; an answer written to find the token in the runner's own frame
/// on purpose is out of reach of any in-process check, and still runs only in the sandbox.
const RUN_TESTS: &str = "import os, sys, unittest\n\
     token = sys.stdin.readline().strip()\nsys.stdin.close()\n\
     d, name, out = sys.argv[1], sys.argv[2], sys.argv[3]\ndel sys.argv[1:]\n\
     sys.path.insert(0, d)\nos.chdir(d)\n\
     kept = [(c, dict(vars(c))) for c in (unittest.TestCase, unittest.TestSuite,\n\
         unittest.TestResult, unittest.TextTestResult, unittest.TextTestRunner)]\n\
     def run():\n\
     \x20   try:\n\
     \x20       suite = unittest.defaultTestLoader.loadTestsFromName(name)\n\
     \x20       for c, attrs in kept:\n\
     \x20           for k in set(vars(c)) - set(attrs):\n\
     \x20               delattr(c, k)\n\
     \x20           for k, v in attrs.items():\n\
     \x20               if vars(c).get(k) is not v:\n\
     \x20                   setattr(c, k, v)\n\
     \x20       return unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(suite)\n\
     \x20   except BaseException:\n\
     \x20       return None\n\
     r = run()\n\
     if r is not None:\n\
     \x20   with open(out, 'w') as f:\n\
     \x20       f.write(' '.join(map(str, [token, r.testsRun, len(r.failures), len(r.errors),\n\
     \x20           len(r.skipped), len(r.expectedFailures), len(r.unexpectedSuccesses)])))\n\
     os._exit(0)\n";

/// Where [`RUN_TESTS`] writes its report, in the work directory.
const RESULT_FILE: &str = "_scr_result";

/// Whether [`RUN_TESTS`]' report says every test ran and passed: this run's `token`, at least one
/// test run, and no failure, error, skip, expected failure or unexpected success (no exercise
/// skips or expects a failure, so one is the answer's doing).
fn reported_pass(report: &str, token: &str) -> bool {
    let mut fields = report.split_whitespace();
    if fields.next() != Some(token) {
        return false;
    }
    let counts: Option<Vec<usize>> = fields.map(|f| f.parse().ok()).collect();
    matches!(counts.as_deref(), Some([run, 0, 0, 0, 0, 0]) if *run > 0)
}

impl TestRunner {
    /// `python` (a path, or a name on `PATH`) under `/usr/bin/sandbox-exec`, once it has run the
    /// sandbox's own test: `unittest` imports under the profile. None where either is missing, or
    /// the interpreter cannot run there, as `/usr/bin/python3` cannot without the Command Line
    /// Tools (an xcrun shim that fails): a cell records no accuracy rather than every answer wrong.
    pub(crate) fn find(python: &str) -> Option<Self> {
        let sandbox = PathBuf::from("/usr/bin/sandbox-exec");
        if !sandbox.exists() {
            return None;
        }
        let out = std::process::Command::new(which(python)?)
            .args(["-I", "-c", PROBE])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())?;
        let text = String::from_utf8(out.stdout).ok()?;
        let mut lines = text.lines();
        let python = PathBuf::from(lines.next()?);
        let home = PathBuf::from(lines.next()?);
        let runner = Self {
            python,
            home,
            sandbox,
        };
        let work = WorkDir::new().ok()?;
        let ok = runner
            .run(&work.0, &["-c", "import unittest"], TEST_TIMEOUT, "")
            .ok()?;
        ok.then_some(runner)
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
        let script = dir.join("_scr_run_tests.py");
        let report = dir.join(RESULT_FILE);
        let (script, work_dir, out) = (
            script.to_string_lossy(),
            dir.to_string_lossy(),
            report.to_string_lossy(),
        );
        // A token this run alone knows, so a report the answer wrote is not the runner's.
        let token = format!("{:032x}", rand::random::<u128>());
        let args = [&*script, &*work_dir, exercise.test_module(), &*out];
        self.run(&dir, &args, limit, &token)
            .with_context(|| format!("running {}'s tests", exercise.name))?;
        let report = std::fs::read_to_string(&report).unwrap_or_default();
        Ok(reported_pass(&report, &token))
    }

    /// Run the interpreter (`-I -B`, then `args`) in `dir` under the sandbox, with `input` on its
    /// stdin, stopped after `limit`: whether it exited 0. It runs in a process group of its own,
    /// its memory watched, and the whole group is killed when it ends or times out, so nothing it
    /// forked outlives it into the next cells' timing.
    fn run(&self, dir: &Path, args: &[&str], limit: Duration, input: &str) -> Result<bool> {
        use std::io::Write;
        use std::os::unix::process::CommandExt;
        let profile = dir.join("_scr.sb");
        std::fs::write(&profile, SANDBOX_PROFILE)?;
        let mut cmd = std::process::Command::new(&self.sandbox);
        cmd.arg("-f")
            .arg(&profile)
            .arg("-D")
            .arg(format!("WORK={}", dir.display()))
            .arg("-D")
            .arg(format!("PYTHON={}", self.python.display()))
            .arg("-D")
            .arg(format!("PYTHON_HOME={}", self.home.display()))
            .arg(&self.python)
            .args(["-I", "-B"])
            .args(args)
            .current_dir(dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .process_group(0);
        let mut child = cmd.spawn()?;
        let group = child.id() as libc::pid_t;
        // One short line, far under the pipe's buffer: written whole before the child reads it.
        // A child that exited first leaves a broken pipe, which is its failure, not the runner's.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = writeln!(stdin, "{input}");
        }
        let start = std::time::Instant::now();
        let passed = loop {
            if let Some(status) = child.try_wait()? {
                break status.success();
            }
            if start.elapsed() > limit || group_footprint(group) > TEST_MEMORY {
                break false;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        // The whole group, done or not: a child it forked is in it.
        // SAFETY: kill(2) with a negative pid signals that process group; it is this child's own.
        unsafe { libc::kill(-group, libc::SIGKILL) };
        let _ = child.wait();
        Ok(passed)
    }
}

/// The physical memory of every process in group `group`, as Activity Monitor counts it (its
/// footprint), summed: what the group takes from the machine.
#[cfg(target_os = "macos")]
fn group_footprint(group: libc::pid_t) -> u64 {
    let mut pids = [0 as libc::pid_t; 256];
    let size = std::mem::size_of_val(&pids) as libc::c_int;
    // SAFETY: the buffer is `size` bytes of pid_t; the call fills at most that many.
    let n = unsafe { libc::proc_listpgrppids(group, pids.as_mut_ptr().cast(), size) };
    let n = (n.max(0) as usize).min(pids.len());
    (pids[..n].iter())
        .filter(|&&pid| pid > 0)
        .map(|&pid| {
            let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
            // SAFETY: a rusage_info_v2 for RUSAGE_INFO_V2, as the call expects.
            let ok = unsafe {
                libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V2, info.as_mut_ptr().cast())
            };
            // SAFETY: zeroed, then filled on success; a failed call leaves zeroes.
            if ok == 0 {
                unsafe { info.assume_init() }.ri_phys_footprint
            } else {
                0
            }
        })
        .sum()
}

/// Off macOS nothing is watched: there is no `sandbox-exec`, so [`TestRunner::find`] finds no
/// runner and no answer's tests run.
#[cfg(not(target_os = "macos"))]
fn group_footprint(_group: libc::pid_t) -> u64 {
    0
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

    /// A right answer, its tests passing, with `escape` run at import: the answer passes only if
    /// the escape raised nothing, so each escape is checked by what it left behind.
    fn right_with(escape: &str) -> String {
        format!(
            "```python\ntry:\n{}\nexcept Exception:\n    pass\ndef add(a, b):\n    return a + b\n```",
            escape
                .lines()
                .map(|l| format!("    {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// An answer passes only on its tests' own report, not on how the run exited: one that ends the
    /// run as it loads (`sys.exit(0)`, `os._exit(0)`, `raise SystemExit`) never ran a test, and
    /// one that switches off `unittest`'s asserts as it loads is tested with them back on.
    #[test]
    fn an_answer_that_exits_or_patches_the_tests_fails() {
        let Some(runner) = TestRunner::find("python3") else {
            eprintln!("skipped: no python3 or sandbox-exec");
            return;
        };
        let e = exercise();
        for (trick, answer) in [
            ("sys.exit(0)", "import sys\nsys.exit(0)"),
            ("os._exit(0)", "import os\nos._exit(0)"),
            ("raise SystemExit", "raise SystemExit"),
            (
                "a wrong add with assertEqual switched off",
                "import unittest\nunittest.TestCase.assertEqual = lambda *a, **k: None\n\
                 def add(a, b):\n    return 0",
            ),
        ] {
            let answer = format!("```python\n{answer}\n```");
            assert!(!runner.passes(&e, &answer).unwrap(), "{trick} passed");
        }
        // The same right answer still passes once the tests are back as they were.
        let right = "```python\nimport unittest\nunittest.TestCase.assertEqual = lambda *a, **k: None\n\
                     def add(a, b):\n    return a + b\n```";
        assert!(runner.passes(&e, right).unwrap());
    }

    /// The sandbox denies by default: an answer can reach no Mach service that writes for it
    /// (`defaults write`, through cfprefsd), exec nothing but the interpreter, submit no launchd
    /// job, and read or stat nothing outside its work directory and Python's own files.
    #[test]
    fn an_answer_reaches_nothing_outside_its_directory() {
        let Some(runner) = TestRunner::find("python3") else {
            eprintln!("skipped: no python3 or sandbox-exec");
            return;
        };
        let e = exercise();
        let domain = format!("com.scr.sandbox-test.{}", std::process::id());
        let read = || {
            std::process::Command::new("/usr/bin/defaults")
                .args(["read", &domain, "k"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        };
        let prefs = right_with(&format!(
            "import subprocess\nsubprocess.run(['/usr/bin/defaults', 'write', '{domain}', 'k', 'x'])"
        ));
        assert!(runner.passes(&e, &prefs).unwrap());
        let wrote = read();
        let _ = std::process::Command::new("/usr/bin/defaults")
            .args(["delete", &domain])
            .output();
        assert!(!wrote, "defaults write reached cfprefsd");

        let marker = std::env::temp_dir().join(format!("scr-sandbox-exec-{}", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let exec = right_with(&format!(
            "import subprocess\nsubprocess.run(['/usr/bin/touch', {:?}])",
            marker.display().to_string()
        ));
        assert!(runner.passes(&e, &exec).unwrap());
        assert!(!marker.exists(), "a binary outside the interpreter ran");

        let label = format!("com.scr.sandbox-test.{}", std::process::id());
        let job = right_with(&format!(
            "import subprocess\nsubprocess.run(['/bin/launchctl', 'submit', '-l', '{label}', '--', '/usr/bin/true'])"
        ));
        assert!(runner.passes(&e, &job).unwrap());
        let submitted = std::process::Command::new("/bin/launchctl")
            .args(["list", &label])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let _ = std::process::Command::new("/bin/launchctl")
            .args(["remove", &label])
            .output();
        assert!(!submitted, "a launchd job was submitted");

        // A read outside the work directory raises: the answer that needs it fails.
        let home = dirs::home_dir().unwrap();
        let reads = format!(
            "```python\nimport os\nos.listdir({:?})\ndef add(a, b):\n    return a + b\n```",
            home.display().to_string()
        );
        assert!(
            !runner.passes(&e, &reads).unwrap(),
            "the home directory was read"
        );
        // Nor can it stat a path there: whether a file exists, its size or its times.
        let stats = format!(
            "```python\nimport os\nos.stat({:?})\ndef add(a, b):\n    return a + b\n```",
            home.join(".ssh").display().to_string()
        );
        assert!(
            !runner.passes(&e, &stats).unwrap(),
            "a path outside was stat'd"
        );
    }

    /// A timed-out answer leaves nothing running: it cannot fork (the profile has no
    /// `process-fork`), and whatever it does start is in its own process group, killed with it,
    /// so nothing outlives it into the next cells' timing.
    #[test]
    fn a_timed_out_answer_leaves_nothing_running() {
        let Some(runner) = TestRunner::find("python3") else {
            eprintln!("skipped: no python3 or sandbox-exec");
            return;
        };
        let e = exercise();
        // An answer that tries to fork, then hangs: its own tag on the command line of anything
        // it leaves, so other tests' runs in parallel are not counted.
        let tag = format!("scr-fork-test-{}", std::process::id());
        let fork = format!(
            "```python\nimport os, sys, time\nsys.argv.append('{tag}')\n\
             try:\n    pid = os.fork()\nexcept OSError:\n    pid = None\n\
             if pid == 0:\n    os.execv(sys.executable, [sys.executable, '-c', 'while True: pass', '{tag}'])\n\
             time.sleep(60)\n```"
        );
        let start = std::time::Instant::now();
        assert!(
            !runner
                .passes_within(&e, &fork, Duration::from_secs(2))
                .unwrap()
        );
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "it was stopped at its limit"
        );
        let left = || {
            std::process::Command::new("/usr/bin/pgrep")
                .args(["-f", &tag])
                .output()
                .map(|o| o.stdout)
                .unwrap_or_default()
        };
        // A SIGKILL lands at once, but the process table lags it a little.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !left().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let left = left();
        assert!(
            String::from_utf8_lossy(&left).trim().is_empty(),
            "something it started outlived it: {}",
            String::from_utf8_lossy(&left)
        );
        // And the fork itself is refused: an answer that needs one fails.
        let forks = "```python\nimport os\nos.fork()\ndef add(a, b):\n    return a + b\n```";
        assert!(!runner.passes(&e, forks).unwrap(), "the answer forked");
    }

    /// An answer that takes far more memory than a test needs is stopped and fails, rather than
    /// taking the Mac's memory from the server the matrix is timing: its group is killed once it
    /// passes [`TEST_MEMORY`], long before 16 GB lands.
    #[test]
    fn an_answer_cannot_take_the_machines_memory() {
        let Some(runner) = TestRunner::find("python3") else {
            eprintln!("skipped: no python3 or sandbox-exec");
            return;
        };
        let e = exercise();
        // A right answer that touches 16 GB a page at a time: stopped part way, it fails.
        let bomb = "```python\nx = bytearray(16 * 1024 ** 3)\n\
                    for i in range(0, len(x), 4096):\n    x[i] = 1\n\
                    def add(a, b):\n    return a + b\n```";
        let start = std::time::Instant::now();
        assert!(!runner.passes(&e, bomb).unwrap(), "stopped, and failed");
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "stopped well before 16 GB"
        );
        // An answer within the cap still passes.
        let modest = "```python\nx = bytearray(256 * 1024 ** 2)\nx[-1] = 1\n\
                      def add(a, b):\n    return a + b\n```";
        assert!(
            runner.passes(&e, modest).unwrap(),
            "256 MB is within the cap"
        );
    }

    /// A Python that cannot run its tests, as `/usr/bin/python3` is without the Command Line
    /// Tools (an xcrun shim that fails), is no test runner: the cell records no accuracy rather
    /// than every answer wrong.
    #[test]
    fn the_matrixs_python_runs_tests_in_the_sandbox() {
        // The matrix passes `--coding-python /usr/bin/python3`: an xcrun shim with the Command Line
        // Tools or Xcode, which the probe resolves to the interpreter it ends in.
        if !std::path::Path::new("/usr/bin/python3").exists() {
            eprintln!("skipped: no /usr/bin/python3");
            return;
        }
        let Some(runner) = TestRunner::find("/usr/bin/python3") else {
            eprintln!("skipped: /usr/bin/python3 cannot run tests here");
            return;
        };
        assert_ne!(
            runner.python,
            std::path::Path::new("/usr/bin/python3"),
            "the shim, resolved"
        );
        let right = "```python\ndef add(a, b):\n    return a + b\n```";
        assert!(runner.passes(&exercise(), right).unwrap());
    }

    #[test]
    fn a_python_that_cannot_run_tests_is_no_runner() {
        let dir = WorkDir::new().unwrap();
        let broken = dir.0.join("python3");
        std::fs::write(&broken, "#!/bin/sh\nexit 1\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&broken, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(TestRunner::find(broken.to_str().unwrap()).is_none());
    }
}
