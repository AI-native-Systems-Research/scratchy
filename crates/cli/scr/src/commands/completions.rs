// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr completions <shell>` — shell completion scripts.
//!
//! Both scripts are GENERATED from clap's own command tree
//! ([`Cli::command`](crate::args::Cli)) at the moment you run
//! `scr completions`. Nothing about the CLI surface is written out by hand here,
//! which matters because that surface is feature-dependent: `serve`, `launch`,
//! `bench`, `batch`, `run-batch` and `model info` are all `#[cfg]`-gated, so a
//! hand-maintained list would advertise commands a given binary doesn't have.
//! The binary that generates the script is the binary being completed, so the
//! subcommands, flags and enum values it emits are exactly right by
//! construction.
//!
//! Model names are the one thing that can't be baked into the script, since they
//! depend on build scope. Those are resolved by calling the binary back: the
//! script shells out to the hidden `model names --source <src>` helper (see
//! [`run_model_names`](crate::commands::model::run_model_names)), which reads a
//! build-time-resolved registry. No network call at completion time.
//!
//! Each argument gets exactly one candidate set, never a union — a model
//! argument completes only ids this build can actually run, and `scr model rm`
//! completes only what is cached. See [`DEFAULT_SOURCE`].

use std::fmt::Write as _;
use std::path::PathBuf;

use clap::{Command, CommandFactory};

use crate::args::{Cli, CompletionShell};

/// Arg ids whose value is a model, and so completes from `model names`.
///
/// Matched on the clap id, not on the command, so a command added later gets
/// model completion for free as long as it names its argument the way every
/// existing one does (`model_tag` positional + `-m/--model`). Deliberately
/// exact: `speculative_model` and friends take a draft model whose candidate set
/// is not this one.
const MODEL_ARG_IDS: &[&str] = &["model", "model_tag"];

/// The candidate set for a model argument: repo ids whose arch, shape and
/// quantization match something this build compiled.
///
/// Deliberately NOT unioned with the local hf-hub cache. The cache is not
/// scoped to the build, so mixing it in offered ids this binary cannot load and
/// made completion look like its scope filter was broken — see
/// `run_model_names`.
const DEFAULT_SOURCE: &str = "compiled";

/// `(command path after `scr`, arg id, source)` — args whose candidate set is
/// NOT [`DEFAULT_SOURCE`]. A table, not logic: each row states a fact about one
/// argument's vocabulary.
///
/// `model rm` is the one that matters. Its argument "must match the model ID
/// shown by `scr model ls`", so completing it from the compiled-in registry
/// would offer ids that are not cached and cannot be removed.
const SOURCE_OVERRIDES: &[(&str, &str, &str)] = &[
    ("model rm", "model", "cached"),
    ("model info", "filters", "stems"),
];

pub fn run_completions(shell: CompletionShell, install: bool) -> anyhow::Result<()> {
    // `build()` is what materializes the auto-generated `--help` / `--version`
    // args and propagates globals. Without it the tree is introspectable but
    // incomplete, and the scripts would omit `--help` everywhere.
    let mut cmd = Cli::command();
    cmd.build();
    let nodes = flatten(&cmd);
    let has_helper = has_names_helper(&cmd);
    let script = match shell {
        CompletionShell::Bash => bash_script(&nodes, has_helper),
        CompletionShell::Zsh => zsh_script(&nodes, has_helper),
    };

    if install {
        install_completion(shell, &script)
    } else {
        print!("{script}");
        Ok(())
    }
}

/// `~/.bashrc` / `~/.zshrc`'s own sourcing block, delimited by a marker so a
/// re-run (after a rebuild that changed the compiled feature set) rewrites the
/// script in place rather than piling up duplicate blocks.
const RC_MARKER_BEGIN: &str = "# >>> scr completions (scr completions --install) >>>";
const RC_MARKER_END: &str = "# <<< scr completions <<<";

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
}

/// Write the script to its standard per-shell location, then make sure the
/// shell's rc file actually loads it, and report exactly what changed.
///
/// bash and zsh need different rc plumbing, which is the whole reason this
/// isn't just "write a file":
///
/// - **bash**: our generated script ends with `complete -F _scr_complete scr`,
///   which runs as soon as the file is sourced — no load-order constraint, so a
///   plain `source <path>` line anywhere in `.bashrc` is sufficient.
/// - **zsh**: the script is an autoloadable `#compdef` function. That only gets
///   picked up by `compinit`'s directory scan, which means the directory it
///   lives in must be on `$fpath` BEFORE `compinit` runs — a `source` line
///   after `compinit` (or anywhere, without the fpath entry) does NOT register
///   it as the completion for `scr`. So this inserts the `fpath=(...)` line
///   before the first existing `compinit` call it finds, rather than simply
///   appending — appending would silently produce a script that looks
///   installed but never completes anything.
fn install_completion(shell: CompletionShell, script: &str) -> anyhow::Result<()> {
    let home = home_dir();
    match shell {
        CompletionShell::Bash => install_bash(&home, script),
        CompletionShell::Zsh => install_zsh(&home, script),
    }
}

/// Takes `home` explicitly (rather than calling [`home_dir`] itself) so tests
/// can point it at a sandboxed directory instead of mutating the real `$HOME`
/// — a global that every test in the binary shares, including ones running
/// concurrently in other threads.
fn install_bash(home: &std::path::Path, script: &str) -> anyhow::Result<()> {
    let script_path = home.join(".local/share/scr/completions.bash");
    write_script(&script_path, script)?;

    let rc_path = home.join(".bashrc");
    let block = format!(
        "{RC_MARKER_BEGIN}\n[ -f \"{path}\" ] && source \"{path}\"\n{RC_MARKER_END}\n",
        path = script_path.display(),
    );
    let rc_changed = ensure_rc_block(&rc_path, &block)?;

    println!("Wrote {}", script_path.display());
    if rc_changed {
        println!(
            "Added a source line to {} (marked, safe to re-run)",
            rc_path.display()
        );
    } else {
        println!(
            "{} already sources it — no change needed",
            rc_path.display()
        );
    }
    println!("Restart your shell, or run: source {}", rc_path.display());
    Ok(())
}

/// See [`install_bash`] for why `home` is a parameter rather than a call to
/// [`home_dir`].
fn install_zsh(home: &std::path::Path, script: &str) -> anyhow::Result<()> {
    let completions_dir = home.join(".zsh/completions");
    let script_path = completions_dir.join("_scr");
    write_script(&script_path, script)?;

    let rc_path = home.join(".zshrc");
    let fpath_line = format!("fpath=({} $fpath)", completions_dir.display());
    let existing = std::fs::read_to_string(&rc_path).unwrap_or_default();
    let had_compinit = existing.lines().any(|l| l.contains("compinit"));

    let rc_changed = if existing.contains(RC_MARKER_BEGIN) {
        false
    } else if let Some(compinit_line) = existing
        .lines()
        .position(|l| l.contains("compinit") && !l.trim_start().starts_with('#'))
    {
        // fpath must land before compinit's directory scan, or the completion
        // is silently never registered — see the module doc above.
        let mut lines: Vec<&str> = existing.lines().collect();
        lines.splice(
            compinit_line..compinit_line,
            [RC_MARKER_BEGIN, fpath_line.as_str(), RC_MARKER_END],
        );
        let mut new_contents = lines.join("\n");
        new_contents.push('\n');
        std::fs::write(&rc_path, new_contents)?;
        true
    } else {
        // No compinit call at all yet — nothing will ever load completions
        // for ANY command until one exists, so add both together.
        let block = format!(
            "{RC_MARKER_BEGIN}\n{fpath_line}\nautoload -Uz compinit && compinit\n{RC_MARKER_END}\n"
        );
        append_block(&rc_path, &block)?;
        true
    };

    println!("Wrote {}", script_path.display());
    if rc_changed {
        let extra = if had_compinit {
            ""
        } else {
            " and a compinit call"
        };
        println!(
            "Updated {} (added fpath entry{extra}, marked, safe to re-run)",
            rc_path.display()
        );
    } else {
        println!(
            "{} already has this on $fpath — no change needed",
            rc_path.display()
        );
    }
    println!("Restart your shell, or run: source {}", rc_path.display());
    Ok(())
}

fn write_script(path: &std::path::Path, script: &str) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, script)?;
    Ok(())
}

/// Ensure `rc_path` contains exactly one copy of `block` (delimited by
/// [`RC_MARKER_BEGIN`]/[`RC_MARKER_END`]), replacing an existing one so a
/// re-run after a rebuild picks up the new script path/content without
/// leaving a stale duplicate behind. Returns whether the file changed.
fn ensure_rc_block(rc_path: &std::path::Path, block: &str) -> anyhow::Result<bool> {
    let existing = std::fs::read_to_string(rc_path).unwrap_or_default();
    if existing.contains(RC_MARKER_BEGIN) {
        let before = existing.split(RC_MARKER_BEGIN).next().unwrap_or_default();
        let after = existing.split(RC_MARKER_END).nth(1).unwrap_or_default();
        // `block` already supplies the newline that separates it from whatever
        // follows; `after` starts with that same newline (it's everything past
        // the END marker LINE), so keeping both would grow the file by one
        // blank line on every re-run.
        let after = after.strip_prefix('\n').unwrap_or(after);
        let rebuilt = format!("{before}{block}{after}");
        if rebuilt == existing {
            return Ok(false);
        }
        std::fs::write(rc_path, rebuilt)?;
        return Ok(true);
    }
    append_block(rc_path, block)?;
    Ok(true)
}

fn append_block(rc_path: &std::path::Path, block: &str) -> anyhow::Result<()> {
    let mut existing = std::fs::read_to_string(rc_path).unwrap_or_default();
    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(block);
    std::fs::write(rc_path, existing)?;
    Ok(())
}

/// One command node, flattened to just what the shell needs.
struct Node {
    /// Path after `scr`, space-joined: `""` for the root, `"model rm"` for a
    /// nested command.
    path: String,
    /// Visible subcommand names.
    subs: Vec<String>,
    /// Every flag spelling, long and short.
    opts: Vec<String>,
    /// The subset of `opts` that consumes a value — where we must NOT offer a
    /// model just because the command accepts one somewhere.
    value_opts: Vec<String>,
    /// Flags whose value is a model, with the source to draw it from.
    model_opts: Vec<(String, String)>,
    /// `(flag, values)` for flags with a closed value set (clap `value_enum`).
    enum_opts: Vec<(String, Vec<String>)>,
    /// Source for a bare positional here, if it takes a model.
    positional: Option<String>,
    /// Closed value set for a bare positional (e.g. `completions <shell>`).
    positional_values: Vec<String>,
}

/// Whether the hidden `model names` helper exists in this build. It rides the
/// optional `model` dependency edge, so a binary built with no model scope has
/// no helper — in which case the script omits model completion entirely rather
/// than shelling out to a command that will fail.
fn has_names_helper(root: &Command) -> bool {
    root.get_subcommands()
        .find(|c| c.get_name() == "model")
        .is_some_and(|m| m.get_subcommands().any(|c| c.get_name() == "names"))
}

fn flatten(root: &Command) -> Vec<Node> {
    let mut out = Vec::new();
    visit(root, String::new(), &mut out);
    out
}

fn visit(cmd: &Command, path: String, out: &mut Vec<Node>) {
    let mut node = Node {
        path: path.clone(),
        subs: Vec::new(),
        opts: Vec::new(),
        value_opts: Vec::new(),
        model_opts: Vec::new(),
        enum_opts: Vec::new(),
        positional: None,
        positional_values: Vec::new(),
    };

    for arg in cmd.get_arguments() {
        if arg.is_hide_set() {
            continue;
        }
        let takes_value = arg
            .get_num_args()
            .map(|range| range.takes_values())
            .unwrap_or_else(|| arg.get_action().takes_values());
        let values: Vec<String> = arg
            .get_possible_values()
            .iter()
            .filter(|v| !v.is_hide_set())
            .map(|v| v.get_name().to_string())
            .collect();
        let source = source_for(&path, arg.get_id().as_str());

        if arg.is_positional() {
            if let Some(source) = source {
                node.positional = Some(source.to_string());
            }
            node.positional_values.extend(values);
            continue;
        }

        let mut spellings = Vec::new();
        if let Some(long) = arg.get_long() {
            spellings.push(format!("--{long}"));
        }
        if let Some(short) = arg.get_short() {
            spellings.push(format!("-{short}"));
        }
        for spelling in spellings {
            node.opts.push(spelling.clone());
            if takes_value {
                node.value_opts.push(spelling.clone());
                if let Some(source) = source {
                    node.model_opts.push((spelling.clone(), source.to_string()));
                } else if !values.is_empty() {
                    node.enum_opts.push((spelling.clone(), values.clone()));
                }
            }
        }
    }

    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        node.subs.push(sub.get_name().to_string());
    }
    out.push(node);

    // Recurse into hidden subcommands too: they are not offered as completions,
    // but a user who types one by hand should still get its flags completed.
    for sub in cmd.get_subcommands() {
        let child = if path.is_empty() {
            sub.get_name().to_string()
        } else {
            format!("{path} {}", sub.get_name())
        };
        visit(sub, child, out);
    }
}

/// The candidate source for one arg, or `None` if it isn't a model argument.
fn source_for(path: &str, id: &str) -> Option<&'static str> {
    if let Some((_, _, source)) = SOURCE_OVERRIDES
        .iter()
        .find(|(p, a, _)| *p == path && *a == id)
    {
        return Some(source);
    }
    MODEL_ARG_IDS.contains(&id).then_some(DEFAULT_SOURCE)
}

/// `|chat|model|model ls|…|` — the set of real command paths, so the script can
/// tell a subcommand from a positional value while walking the words the user
/// has typed. Without it, `scr chat granite serve` would parse `serve` as a
/// command.
fn paths_literal(nodes: &[Node]) -> String {
    let mut s = String::from("|");
    for node in nodes.iter().filter(|n| !n.path.is_empty()) {
        let _ = write!(s, "{}|", node.path);
    }
    s
}

/// The per-path `case` arms, shared verbatim by both scripts — plain POSIX
/// `case` and scalar assignments, which bash and zsh read identically. Only the
/// completion primitives around them differ.
fn case_arms(nodes: &[Node], has_helper: bool) -> String {
    let mut s = String::new();
    for node in nodes {
        let pattern = if node.path.is_empty() {
            "\"\"".to_string()
        } else {
            format!("\"{}\"", node.path)
        };
        let _ = writeln!(s, "        {pattern})");
        if !node.subs.is_empty() {
            let _ = writeln!(s, "            subs=\"{}\"", node.subs.join(" "));
        }
        if !node.opts.is_empty() {
            let _ = writeln!(s, "            opts=\"{}\"", node.opts.join(" "));
        }
        if !node.value_opts.is_empty() {
            let _ = writeln!(
                s,
                "            value_opts=\"{}\"",
                node.value_opts.join(" ")
            );
        }
        if !node.enum_opts.is_empty() {
            let mut map = String::from("|");
            for (flag, values) in &node.enum_opts {
                let _ = write!(map, "{flag}:{}|", values.join(" "));
            }
            let _ = writeln!(s, "            enum_map=\"{map}\"");
        }
        if !node.positional_values.is_empty() {
            let _ = writeln!(
                s,
                "            positional_values=\"{}\"",
                node.positional_values.join(" ")
            );
        }
        // Model completion needs the helper to call. Without it, emit nothing
        // rather than a call that would fail on every TAB.
        if has_helper {
            if !node.model_opts.is_empty() {
                let flags: Vec<&str> = node.model_opts.iter().map(|(f, _)| f.as_str()).collect();
                let _ = writeln!(s, "            model_opts=\"{}\"", flags.join(" "));
                let _ = writeln!(s, "            model_source=\"{}\"", node.model_opts[0].1);
            }
            if let Some(source) = &node.positional {
                let _ = writeln!(s, "            positional=\"{source}\"");
            }
        }
        let _ = writeln!(s, "            ;;");
    }
    s
}

/// The shared preamble: resolve the command path from the words typed so far by
/// greedily extending it while the result is still a real command path.
fn path_walk(
    paths: &str,
    first_word: usize,
    cursor: &str,
    words: &dyn Fn(&str) -> String,
) -> String {
    format!(
        r#"    local paths="{paths}"
    path=""
    i={first_word}
    while [ "$i" -lt "${cursor}" ]; do
        w={word_i}
        case "$w" in
            -*) ;;
            *)
                candidate="${{path:+$path }}$w"
                case "$paths" in
                    *"|$candidate|"*) path="$candidate" ;;
                    *) break ;;
                esac
                ;;
        esac
        i=$((i + 1))
    done
"#,
        word_i = words("i"),
    )
}

fn bash_script(nodes: &[Node], has_helper: bool) -> String {
    let paths = paths_literal(nodes);
    let arms = case_arms(nodes, has_helper);
    let walk = path_walk(&paths, 1, "COMP_CWORD", &|i| {
        format!("\"${{COMP_WORDS[{i}]}}\"")
    });
    format!(
        r#"# scr(1) bash completion                            -*- shell-script -*-
#
# Generated by `scr completions bash` from this binary's own clap command tree,
# so the subcommands and flags below are exactly the ones it was compiled with.
# Regenerate after rebuilding with a different feature set.
#
# Install: scr completions bash > /usr/local/etc/bash_completion.d/scr
#          (or source it from ~/.bashrc)
_scr_complete() {{
    local cur prev scr path candidate w i vals
    local subs opts value_opts model_opts model_source positional positional_values enum_map
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev=""
    [ "$COMP_CWORD" -gt 0 ] && prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    # Call back into the binary the user actually invoked, not a bare `scr`:
    # completions must work for a binary that is not on PATH under that name.
    scr="${{COMP_WORDS[0]}}"
    COMPREPLY=()

{walk}
    subs=""; opts=""; value_opts=""; model_opts=""; model_source=""
    positional=""; positional_values=""; enum_map=""
    case "$path" in
{arms}    esac

    # A flag whose value is a model.
    if [ -n "$model_opts" ] && [[ " $model_opts " == *" $prev "* ]]; then
        COMPREPLY=( $(compgen -W "$("$scr" model names --source "$model_source" -- "$cur" 2>/dev/null)" -- "$cur") )
        return
    fi
    # A flag with a closed value set (clap value_enum).
    if [ -n "$enum_map" ] && [[ "$enum_map" == *"|$prev:"* ]]; then
        vals="${{enum_map#*|$prev:}}"
        vals="${{vals%%|*}}"
        COMPREPLY=( $(compgen -W "$vals" -- "$cur") )
        return
    fi
    # Any other flag that wants a value: its vocabulary is open (a port, a path,
    # a dtype), so offer nothing and let `complete -o default` fall back to
    # filenames — which is what most of them actually want.
    if [ -n "$value_opts" ] && [[ " $value_opts " == *" $prev "* ]]; then
        return
    fi
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return
    fi
    if [ -n "$subs" ] || [ -n "$positional_values" ]; then
        COMPREPLY=( $(compgen -W "$subs $positional_values" -- "$cur") )
    fi
    if [ -n "$positional" ]; then
        COMPREPLY+=( $(compgen -W "$("$scr" model names --source "$positional" -- "$cur" 2>/dev/null)" -- "$cur") )
    fi
}}
complete -o default -F _scr_complete scr
"#
    )
}

fn zsh_script(nodes: &[Node], has_helper: bool) -> String {
    let paths = paths_literal(nodes);
    let arms = case_arms(nodes, has_helper);
    let walk = path_walk(&paths, 2, "CURRENT", &|i| format!("\"${{words[{i}]}}\""));
    format!(
        r#"#compdef scr
# scr(1) zsh completion
#
# Generated by `scr completions zsh` from this binary's own clap command tree,
# so the subcommands and flags below are exactly the ones it was compiled with.
# Regenerate after rebuilding with a different feature set.
#
# Install: scr completions zsh > ~/.zsh/completions/_scr   (dir must be on $fpath)
_scr() {{
    local cur prev scr path candidate w i vals
    local subs opts value_opts model_opts model_source positional positional_values enum_map
    cur="${{words[CURRENT]}}"
    prev=""
    [ "$CURRENT" -gt 1 ] && prev="${{words[CURRENT-1]}}"
    # Call back into the binary the user actually invoked, not a bare `scr`.
    scr="${{words[1]}}"

{walk}
    subs=""; opts=""; value_opts=""; model_opts=""; model_source=""
    positional=""; positional_values=""; enum_map=""
    case "$path" in
{arms}    esac

    if [ -n "$model_opts" ] && [[ " $model_opts " == *" $prev "* ]]; then
        compadd -- ${{(f)"$("$scr" model names --source "$model_source" -- "$cur" 2>/dev/null)"}}
        return
    fi
    if [ -n "$enum_map" ] && [[ "$enum_map" == *"|$prev:"* ]]; then
        vals="${{enum_map#*|$prev:}}"
        vals="${{vals%%|*}}"
        compadd -- ${{=vals}}
        return
    fi
    # Open-vocabulary flag value: offer files, the usual intent.
    if [ -n "$value_opts" ] && [[ " $value_opts " == *" $prev "* ]]; then
        _files
        return
    fi
    if [[ "$cur" == -* ]]; then
        compadd -- ${{=opts}}
        return
    fi
    [ -n "$subs" ] && compadd -- ${{=subs}}
    [ -n "$positional_values" ] && compadd -- ${{=positional_values}}
    [ -n "$positional" ] && compadd -- ${{(f)"$("$scr" model names --source "$positional" -- "$cur" 2>/dev/null)"}}
}}
_scr "$@"
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built() -> Command {
        let mut cmd = Cli::command();
        cmd.build();
        cmd
    }

    fn script(shell: CompletionShell) -> String {
        let cmd = built();
        let nodes = flatten(&cmd);
        let has_helper = has_names_helper(&cmd);
        match shell {
            CompletionShell::Bash => bash_script(&nodes, has_helper),
            CompletionShell::Zsh => zsh_script(&nodes, has_helper),
        }
    }

    /// The whole point of generating from clap: the offered subcommands are the
    /// ones THIS build has. A hardcoded list would name `serve` in a chat-only
    /// binary (and miss it in a `-Fserve` one).
    #[test]
    fn offers_exactly_this_builds_subcommands() {
        let cmd = built();
        let visible: Vec<String> = cmd
            .get_subcommands()
            .filter(|c| !c.is_hide_set())
            .map(|c| c.get_name().to_string())
            .collect();
        assert!(
            visible.iter().any(|c| c == "chat"),
            "chat is always compiled in"
        );

        for shell in [CompletionShell::Bash, CompletionShell::Zsh] {
            let s = script(shell);
            // The root arm's list. Skipped: the `subs=""; opts=""; ...` reset
            // line, which also starts with `subs="`.
            let subs_line = s
                .lines()
                .find(|l| l.trim_start().starts_with("subs=\"") && !l.contains("opts="))
                .expect("root arm sets subs");
            for name in &visible {
                assert!(
                    subs_line.contains(name.as_str()),
                    "{name} missing from root completions: {subs_line}"
                );
            }
            // Anything NOT compiled in must be absent. `serve` is the load-bearing
            // case: it is `#[cfg(feature = "serve")]`, and tests run without it.
            if !visible.iter().any(|c| c == "serve") {
                assert!(
                    !subs_line.contains("serve"),
                    "advertised `serve` in a build without it: {subs_line}"
                );
            }
        }
    }

    /// `model names` is `hide = true` — it exists for the scripts to call, and
    /// must never be offered to a user as a completion. It still gets a `case`
    /// arm (so its own flags complete if typed by hand), hence the check is on
    /// the offered lists, not on the whole script.
    #[test]
    fn never_offers_the_hidden_helper() {
        for shell in [CompletionShell::Bash, CompletionShell::Zsh] {
            let s = script(shell);
            let offered = s
                .lines()
                .filter(|l| l.trim_start().starts_with("subs=\"") && !l.contains("opts="));
            for line in offered {
                assert!(
                    !line
                        .split_whitespace()
                        .any(|w| w.trim_matches('"') == "names"),
                    "offered the hidden helper: {line}"
                );
            }
        }
    }

    /// Guards the arg walk: chat's flags, its `-m/--model` model source, and its
    /// positional model must all land in the generated arm.
    #[test]
    fn chat_arm_carries_flags_and_model_sources() {
        let s = script(CompletionShell::Bash);
        let arm = s
            .split("\"chat\")")
            .nth(1)
            .expect("chat arm")
            .split(";;")
            .next()
            .unwrap();
        assert!(arm.contains("--device"), "flags missing: {arm}");
        assert!(arm.contains("--help"), "clap's own args missing: {arm}");
        assert!(arm.contains("model_opts=\"--model -m\""), "{arm}");
        assert!(arm.contains("positional=\"compiled\""), "{arm}");
    }

    /// `model rm` takes an id that "must match the model ID shown by `scr model
    /// ls`", so it completes from the cache, never from the compiled registry.
    #[test]
    fn model_rm_completes_from_the_cache_only() {
        let s = script(CompletionShell::Bash);
        let arm = s
            .split("\"model rm\")")
            .nth(1)
            .expect("model rm arm")
            .split(";;")
            .next()
            .unwrap();
        assert!(arm.contains("positional=\"cached\""), "{arm}");
        assert!(!arm.contains("\"compiled\""), "{arm}");
    }

    /// A binary built without any model scope has no `model names` subcommand
    /// (it rides the optional `model` dependency edge — this is the shape CI's
    /// `--workspace --features cuda,scratchy-models/all` produces). The script
    /// must then omit model completion outright rather than emit TAB handlers
    /// that shell out to a command that does not exist.
    #[test]
    fn omits_model_completion_without_the_helper() {
        let cmd = built();
        let nodes = flatten(&cmd);
        let arms = case_arms(&nodes, false);
        assert!(!arms.contains("model_opts="), "{arms}");
        assert!(!arms.contains("positional=\""), "{arms}");
        // The rest of the surface is unaffected.
        assert!(arms.contains("--device"), "{arms}");
        assert!(arms.contains("positional_values=\"bash zsh\""), "{arms}");
    }

    /// Regression test for the report that motivated dropping the union: a
    /// `-Fmodel/llama-3.2-3b,quant/mlx` build offered cached granite
    /// checkpoints, non-MLX checkpoints and `modernbert-embed-base` (not even a
    /// decoder), and — with the registry empty — the SAME list for every
    /// `-Fmodel` scope. The cause was model args sourcing `registry ∪ cache`.
    ///
    /// No model argument may draw from the unscoped local cache. `model rm` is
    /// the sole `cached` consumer, and its value is not a model to run.
    #[test]
    fn model_arguments_never_source_the_unscoped_cache() {
        let cmd = built();
        let nodes = flatten(&cmd);

        for node in &nodes {
            let model_sources = node
                .model_opts
                .iter()
                .map(|(_, source)| source.as_str())
                .chain(node.positional.as_deref())
                .collect::<Vec<_>>();
            for source in model_sources {
                let is_rm_or_info = node.path == "model rm" || node.path == "model info";
                assert!(
                    source == DEFAULT_SOURCE || is_rm_or_info,
                    "`{}` completes a model from `{source}`, which is not build-scoped",
                    node.path
                );
            }
        }

        // And the emitted scripts must never name a union-ish source.
        for shell in [CompletionShell::Bash, CompletionShell::Zsh] {
            let s = script(shell);
            assert!(
                !s.contains("--source default") && !s.contains("source=\"default\""),
                "generated script still references the removed `default` source"
            );
        }
    }

    fn shell_available(shell: &str) -> bool {
        std::process::Command::new(shell)
            .arg("-c")
            .arg("exit 0")
            .status()
            .is_ok_and(|s| s.success())
    }

    fn syntax_check(shell: &str, script_text: &str) {
        if !shell_available(shell) {
            eprintln!("skipping: {shell} not available");
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("completion");
        std::fs::write(&path, script_text).expect("write script");
        let out = std::process::Command::new(shell)
            .arg("-n")
            .arg(&path)
            .output()
            .expect("run shell");
        assert!(
            out.status.success(),
            "{shell} rejected the generated script:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Generated shell is only as good as its quoting, and a syntax error is
    /// invisible until someone presses TAB.
    #[test]
    fn generated_scripts_are_syntactically_valid() {
        syntax_check("bash", &script(CompletionShell::Bash));
        syntax_check("zsh", &script(CompletionShell::Zsh));
    }

    /// The regression that motivated the rewrite: the prototype offered model
    /// ids after every flag, so `--device <TAB>` suggested models. A
    /// value-taking flag with an open vocabulary must yield nothing and let
    /// bash fall back to filenames.
    #[test]
    fn does_not_offer_models_as_an_arbitrary_flag_value() {
        if !shell_available("bash") {
            eprintln!("skipping: bash not available");
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let completion = dir.path().join("completion.bash");
        std::fs::write(&completion, script(CompletionShell::Bash)).expect("write");

        // A stub `scr` so a model suggestion, if one were offered, would be
        // unmistakable — and so the test doesn't depend on a built binary.
        let stub = dir.path().join("scr");
        std::fs::write(&stub, "#!/bin/sh\necho stub-org/stub-model\n").expect("write stub");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }

        let harness = format!(
            r#"source '{completion}'
            probe() {{
                COMP_WORDS=('{stub}' "$@" "")
                COMP_CWORD=$(( ${{#COMP_WORDS[@]}} - 1 ))
                COMPREPLY=()
                _scr_complete
                echo "${{COMPREPLY[*]}}"
            }}
            echo "device:$(probe chat --device)"
            echo "model:$(probe chat -m)"
            echo "positional:$(probe chat)"
            "#,
            completion = completion.display(),
            stub = stub.display(),
        );
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(&harness)
            .output()
            .expect("run bash");
        let stdout = String::from_utf8_lossy(&out.stdout);

        assert!(
            stdout.contains("device:\n") || stdout.contains("device:"),
            "no device line: {stdout}"
        );
        let device_line = stdout
            .lines()
            .find(|l| l.starts_with("device:"))
            .expect("device line");
        assert_eq!(
            device_line.trim(),
            "device:",
            "offered a value for --device, which has an open vocabulary: {stdout}"
        );
        // And the flag/positional that DO take a model still complete.
        for prefix in ["model:", "positional:"] {
            let line = stdout
                .lines()
                .find(|l| l.starts_with(prefix))
                .unwrap_or_else(|| panic!("no {prefix} line: {stdout}"));
            assert!(
                line.contains("stub-org/stub-model"),
                "model completion broken for {prefix}: {stdout}"
            );
        }
    }

    /// `ensure_rc_block` must be byte-stable across repeated installs — this is
    /// a regression test for a bug caught during manual testing: the first
    /// implementation grew the rc file by one blank line on every re-run,
    /// because the text captured "after the END marker" already started with
    /// the newline the block itself supplies.
    #[test]
    fn ensure_rc_block_is_byte_stable_across_reinstalls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rc_path = dir.path().join("rc");
        std::fs::write(&rc_path, "# existing content\nalias g=git\n").expect("seed rc");
        let block = format!("{RC_MARKER_BEGIN}\nsource /some/path\n{RC_MARKER_END}\n");

        assert!(ensure_rc_block(&rc_path, &block).expect("first install"));
        let after_first = std::fs::read_to_string(&rc_path).expect("read");

        for _ in 0..3 {
            let changed = ensure_rc_block(&rc_path, &block).expect("reinstall");
            assert!(!changed, "reported a change on an unchanged block");
            let contents = std::fs::read_to_string(&rc_path).expect("read");
            assert_eq!(
                contents, after_first,
                "rc file grew or changed on a no-op reinstall"
            );
        }

        assert!(after_first.starts_with("# existing content\nalias g=git\n"));
        assert_eq!(after_first.matches(RC_MARKER_BEGIN).count(), 1);
    }

    /// Rebuilding with a different feature set changes the script content
    /// (e.g. a newly-compiled model or a newly-`#[cfg]`-gated subcommand), so a
    /// re-run must replace the block in place, not leave the stale one behind.
    #[test]
    fn ensure_rc_block_replaces_a_changed_block_in_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rc_path = dir.path().join("rc");
        std::fs::write(&rc_path, "alias g=git\n").expect("seed rc");

        let old_block = format!("{RC_MARKER_BEGIN}\nsource /old/path\n{RC_MARKER_END}\n");
        let new_block = format!("{RC_MARKER_BEGIN}\nsource /new/path\n{RC_MARKER_END}\n");

        ensure_rc_block(&rc_path, &old_block).expect("first install");
        let changed = ensure_rc_block(&rc_path, &new_block).expect("update");
        assert!(changed);

        let contents = std::fs::read_to_string(&rc_path).expect("read");
        assert!(contents.contains("/new/path"));
        assert!(!contents.contains("/old/path"));
        assert_eq!(contents.matches(RC_MARKER_BEGIN).count(), 1);
    }

    /// `install_bash` end to end against a sandboxed `$HOME`: writes the
    /// script, wires `.bashrc`, and is idempotent.
    #[test]
    fn install_bash_writes_script_and_wires_bashrc_idempotently() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script_path = dir.path().join(".local/share/scr/completions.bash");
        let rc_path = dir.path().join(".bashrc");
        std::fs::write(&rc_path, "alias ll='ls -la'\n").expect("seed rc");

        write_script(&script_path, "# a completion script\n").expect("write");
        let block = format!(
            "{RC_MARKER_BEGIN}\n[ -f \"{p}\" ] && source \"{p}\"\n{RC_MARKER_END}\n",
            p = script_path.display()
        );
        assert!(ensure_rc_block(&rc_path, &block).expect("install"));
        assert!(
            !ensure_rc_block(&rc_path, &block).expect("reinstall"),
            "not idempotent"
        );

        let rc = std::fs::read_to_string(&rc_path).expect("read");
        assert!(
            rc.contains("alias ll='ls -la'"),
            "clobbered existing content"
        );
        assert!(rc.contains(&script_path.display().to_string()));
        assert_eq!(
            std::fs::read_to_string(&script_path).unwrap(),
            "# a completion script\n"
        );
    }

    /// The zsh fpath entry MUST land before an existing `compinit` call — after
    /// it, `compinit`'s directory scan has already run and the entry is
    /// registered too late to be picked up until the next full re-scan. This
    /// is the load-bearing correctness property of `install_zsh`.
    #[test]
    fn zsh_install_places_fpath_before_existing_compinit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rc_path = dir.path().join(".zshrc");
        std::fs::write(
            &rc_path,
            "export EDITOR=vim\nautoload -Uz compinit\ncompinit\nsetopt AUTO_CD\n",
        )
        .expect("seed rc");

        install_zsh(dir.path(), "#compdef scr\n_scr() { :; }\n_scr \"$@\"\n").expect("install");

        let rc = std::fs::read_to_string(&rc_path).expect("read");
        let fpath_pos = rc.find("fpath=(").expect("fpath line present");
        let compinit_pos = rc.rfind("\ncompinit\n").expect("compinit call present");
        assert!(
            fpath_pos < compinit_pos,
            "fpath must precede compinit:\n{rc}"
        );
        // Existing content is preserved, not replaced.
        assert!(rc.contains("setopt AUTO_CD"));
    }

    /// No `compinit` call anywhere yet: nothing would ever load completions for
    /// ANY command, so `install_zsh` must add one alongside the fpath entry
    /// rather than silently writing a script that can never be registered.
    #[test]
    fn zsh_install_adds_compinit_when_none_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rc_path = dir.path().join(".zshrc");
        std::fs::write(&rc_path, "export EDITOR=vim\n").expect("seed rc");

        install_zsh(dir.path(), "#compdef scr\n_scr() { :; }\n_scr \"$@\"\n").expect("install");

        let rc = std::fs::read_to_string(&rc_path).expect("read");
        assert!(rc.contains("compinit"), "no compinit added:\n{rc}");
        assert!(rc.contains("fpath=("));
    }
}
