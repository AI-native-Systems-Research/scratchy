#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Generate the chat-template goldens from HuggingFace transformers.

Every case is rendered through transformers' own chat-template environment
(`chat_template_utils._compile_jinja_template`, the one `apply_chat_template`
uses), so the expected bytes are HuggingFace's, never hand-written. Run with
any Python that has `transformers` (and so `jinja2`) installed:

    python3 scripts/generate_chat_template_goldens.py

Output: `crates/serving/api/src/chat_template/hf_renders.rs`: the `tojson`
filter's `(template, tools JSON, render)` table and the statements' (loop
controls, `{% generation %}`) `(template, messages JSON, render)` table, each
with the templates transformers refuses to render. Needs `rustfmt` on PATH.
"""

import json
import subprocess
from pathlib import Path

import transformers
from transformers.utils.chat_template_utils import _compile_jinja_template

OUT = (
    Path(__file__).resolve().parent.parent
    / "crates/serving/api/src/chat_template/hf_renders.rs"
)

# Every escape class `json.dumps` distinguishes: HTML characters it must NOT
# escape, the short escapes, other C0 controls, DEL, non-ASCII in and beyond
# the BMP, and U+2028.
TEXT = (
    "the user's city, <b> & \"quoted\" back\\slash /path tab\t nl\n cr\r bell"
    + chr(7) + " del" + chr(0x7F) + " " + chr(0xE9) + " " + chr(0x4E2D)
    + " " + chr(0x1F600) + " ff" + chr(0xC) + " bs" + chr(8) + " lsep" + chr(0x2028)
)
TOOLS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": TEXT,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer", "minimum": 1, "maximum": 14},
                    },
                    "required": ["city"],
                },
            },
        }
    ],
    ensure_ascii=False,
)
NESTED = (
    '{"zeta": 1, "Alpha": [], "beta": {}, "c": [1, [2, {"d": null, "e": true, '
    '"f": false}]], "' + chr(0xE9) + '": "x", "Z": {"y": 1, "x": 2}}'
)
NUMBERS = (
    '{"ints": [0, -1, 9223372036854775807, 18446744073709551615, '
    '-9223372036854775808], "floats": [1.0, -0.0, 0.1, 0.00001, 0.0001, 1e16, '
    "1e15, 123456789012345678.0, 1e22, 5e-324, 1.7976931348623157e308, 2.5, "
    "-3.75e-7, 100.0, 1.5e300]}"
)

TOJSON_RENDERS = [
    ("{{ tools | tojson }}", TOOLS),
    ("{{ tools | tojson(ensure_ascii=true) }}", TOOLS),
    ("{{ tools | tojson(true) }}", TOOLS),
    ("{{ tools | tojson(indent=2) }}", NESTED),
    ("{{ tools | tojson(indent=0) }}", NESTED),
    ("{{ tools | tojson(indent=-1) }}", NESTED),
    ("{{ tools | tojson(indent=true) }}", NESTED),
    ("{{ tools | tojson(indent=false) }}", NESTED),
    ('{{ tools | tojson(indent="--") }}', NESTED),
    ('{{ tools | tojson(separators=(",", ":")) }}', NESTED),
    ('{{ tools | tojson(indent=2, separators=(", ", " = ")) }}', NESTED),
    ("{{ tools | tojson(sort_keys=true) }}", NESTED),
    ("{{ tools | tojson(false, 2, none, true) }}", NESTED),
    ("{{ tools | tojson(ensure_ascii=true, sort_keys=true) }}", NESTED),
    ("{{ tools | tojson }}", NUMBERS),
    ("{{ tools | tojson(indent=1) }}", NUMBERS),
    ('{{ {3: "a", 2.5: "b", false: "c", none: "d"} | tojson }}', "[]"),
    ('{{ {3: "a", 1: "b", 2.5: "c"} | tojson(sort_keys=true) }}', "[]"),
    ('{{ (1, "a", none) | tojson }}', "[]"),
    ("{{ 2.0 | tojson }}", "[]"),
    ("{{ (1 / 3) | tojson }}", "[]"),
    ("{{ 1e100 | tojson }}", "[]"),
    ("{{ (1e308 * 10) | tojson }}", "[]"),
    ("{{ (-1e308 * 10) | tojson }}", "[]"),
    ('{{ "a" | tojson }}', "[]"),
    ("{{ none | tojson }}", "[]"),
    ("{{ true | tojson }}", "[]"),
    ("{{ [] | tojson(indent=2) }}", "[]"),
]

TOJSON_ERRORS = [
    ("{{ tools | tojson(foo=1) }}", NESTED),
    ("{{ undefined_name | tojson }}", NESTED),
    ("{{ tools | tojson(true, ensure_ascii=true) }}", NESTED),
    ("{{ tools | tojson(false, none, none, false, 1) }}", NESTED),
    ("{{ tools | tojson(indent=1.5) }}", NESTED),
    ('{{ tools | tojson(separators=(",",)) }}', NESTED),
    ('{{ tools | tojson(separators=(",", 1)) }}', NESTED),
    ('{{ {"a": 1, 2: "b"} | tojson(sort_keys=true) }}', NESTED),
]

# The statements HuggingFace's environment adds to Jinja: `{% break %}` /
# `{% continue %}` (`jinja2.ext.loopcontrols`) and `{% generation %}`
# (`AssistantTracker`, a call block: its body is a caller with its own scope).
MESSAGES = json.dumps(
    [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "tool", "content": "t"},
        {"role": "user", "content": "bye"},
    ]
)
STATEMENT_RENDERS = [
    (
        "{% for m in messages %}{% if m.role == 'tool' %}{% break %}{% endif %}"
        "{{ m.role }}|{% endfor %}",
        MESSAGES,
    ),
    (
        "{% for m in messages %}{% if m.role == 'system' %}{% continue %}{% endif %}"
        "{{ m.role }}|{% endfor %}",
        MESSAGES,
    ),
    (
        "{%- for m in messages %}\n"
        "    {%- if loop.index > 2 %}\n"
        "        {%- break %}\n"
        "    {%- endif %}\n"
        "{{ m.role }}\n"
        "{% endfor %}",
        MESSAGES,
    ),
    (
        "{% for m in messages %}[{% for c in m.content %}{% if c == 'l' %}{% break %}"
        "{% endif %}{{ c }}{% endfor %}]{% endfor %}",
        MESSAGES,
    ),
    (
        "{% for m in messages %}{% if loop.first %}{% continue %}{% endif %}"
        "{{ loop.index }}{% if not loop.last %},{% endif %}{% endfor %}",
        MESSAGES,
    ),
    (
        "{% for m in messages %}{% if m.role == 'assistant' %}{% generation %}"
        "{{ m.content }}{% endgeneration %}{% else %}{{ m.role }}|{% endif %}{% endfor %}",
        MESSAGES,
    ),
    (
        "{%- for m in messages %}\n"
        '{%- if m.role == "assistant" %}\n'
        "{% generation -%}\n"
        "<|assistant|>{{ m.content }}\n"
        "{%- endgeneration %}\n"
        "{% else %}\n"
        "{{ m.role }}: {{ m.content }}\n"
        "{% endif %}\n"
        "{%- endfor %}",
        MESSAGES,
    ),
    (
        "{% set x = 'outer' %}{% generation %}{% set x = 'inner' %}{{ x }}"
        "{% endgeneration %}|{{ x }}",
        MESSAGES,
    ),
    (
        "{% for m in messages %}{% generation %}{{ loop.index }}{{ m.role[0] }}"
        "{% endgeneration %}{% endfor %}",
        MESSAGES,
    ),
    (
        "{% set ns = namespace(n=0) %}{% for m in messages %}{% generation %}"
        "{% set ns.n = ns.n + 1 %}{% endgeneration %}{% endfor %}{{ ns.n }}",
        MESSAGES,
    ),
    (
        '{{ "{% generation %}" }}|{# {% generation %} #}|'
        "{% raw %}{% generation %}{% endraw %}",
        MESSAGES,
    ),
    ("{%generation%}x{%endgeneration%}", MESSAGES),
]
STATEMENT_ERRORS = [
    ("{% generation %}x", MESSAGES),
    ("{% endgeneration %}", MESSAGES),
    ("{% generation foo %}x{% endgeneration %}", MESSAGES),
]


def rust_str(s):
    """A Rust string literal for `s`; controls and line separators escaped."""
    out = ['"']
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F or ch in (chr(0x2028), chr(0x2029)):
            out.append("\\u{%x}" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def render(template, variable, value_json):
    return _compile_jinja_template(template).render(**{variable: json.loads(value_json)})


def emit_renders(out, name, variable, cases):
    out.append(f"/// `(template, {variable} JSON, render)`.")
    out.append(f"pub(super) const {name}: &[(&str, &str, &str)] = &[")
    for template, value in cases:
        rendered = render(template, variable, value)
        out.append(f"    ({rust_str(template)}, {rust_str(value)}, {rust_str(rendered)}),")
    out += ["];", ""]


def emit_errors(out, name, variable, cases):
    out.append(f"/// `(template, {variable} JSON)` that transformers refuses to render,")
    out.append("/// each under the Python exception it raises.")
    out.append(f"pub(super) const {name}: &[(&str, &str)] = &[")
    for template, value in cases:
        try:
            rendered = render(template, variable, value)
        except Exception as exc:  # whatever Python raises is the golden
            out.append(f"    // {type(exc).__name__}: {exc}")
            out.append(f"    ({rust_str(template)}, {rust_str(value)}),")
        else:
            raise SystemExit(f"{template!r} rendered {rendered!r}; expected an error")
    out += ["];", ""]


def main():
    out = [
        "// SPDX-License-Identifier: Apache-2.0",
        "// Copyright contributors to the vLLM project",
        "",
        "// @generated by scripts/generate_chat_template_goldens.py — do not edit.",
        "",
        f"//! Chat-template goldens: transformers {transformers.__version__}'s own renders.",
        "",
    ]
    emit_renders(out, "HF_TOJSON_RENDERS", "tools", TOJSON_RENDERS)
    emit_errors(out, "HF_TOJSON_ERRORS", "tools", TOJSON_ERRORS)
    emit_renders(out, "HF_STATEMENT_RENDERS", "messages", STATEMENT_RENDERS)
    emit_errors(out, "HF_STATEMENT_ERRORS", "messages", STATEMENT_ERRORS)
    OUT.write_text("\n".join(out))
    subprocess.run(["rustfmt", "--edition", "2024", str(OUT)], check=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
