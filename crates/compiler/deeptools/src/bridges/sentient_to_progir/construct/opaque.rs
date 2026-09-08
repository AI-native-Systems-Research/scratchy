//! THE OPAQUE-TEMPLATE INSTRUCTION — `ConstructOpaqueInstr` and the string trim it uses.
//!
//! ⭐ THE TEMPLATES ARE DATA, IN `dcc/src/Conversion/SentientToProgIR/opaqueTemplates/` — 38
//! `.smc` files (exp, gelu, sigmoid, rsqrt, layernormscale, idx32toaddr, …). They are a
//! declared-data table, not code; this crate's `build.rs` ALREADY reads `.smc` mnemonics and
//! emits `InstOpCode::…`, so a template naming a non-opcode fails to compile.
//!
//! 2 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e004_rtrim` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3859` |
//! | `e063_ConstructOpaqueInstr` | 1 | 167 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3866` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e004_rtrim
///
/// One template line with its trailing whitespace dropped.
///
/// ⛔⛔ C'S `isspace` SET, WHICH NEITHER RUST SPELLING MATCHES. `str::trim_end` uses Unicode
/// `White_Space` and so also eats U+00A0 and U+2028; `u8::is_ascii_whitespace` omits `\v` (0x0B),
/// which `std::isspace` includes. A template line ending in a vertical tab would keep it under one
/// and lose a non-breaking space under the other.
#[must_use]
pub fn rtrim(s: &str) -> &str {
    s.trim_end_matches(|ch: char| ch == ' ' || ('\u{9}'..='\u{d}').contains(&ch))
}

// crustify:todo: e063_ConstructOpaqueInstr

#[cfg(test)]
mod unit_tests {
    use super::rtrim;

    /// ⛔ THE TAIL ONLY, AND C'S SET EXACTLY — the vertical tab goes, the non-breaking space stays.
    #[test]
    fn trims_cs_whitespace_off_the_tail_only() {
        assert_eq!(
            rtrim("  SFP_IMMCOPY imm:0 \t\r\n\u{b}\u{c}"),
            "  SFP_IMMCOPY imm:0"
        );
        assert_eq!(rtrim("PE_NOP"), "PE_NOP");
        assert_eq!(rtrim("   "), "");
        assert_eq!(rtrim("PE_NOP\u{a0}"), "PE_NOP\u{a0}");
    }
}
