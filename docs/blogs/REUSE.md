# What "Reuse" Means in the Age of AI

Every generation of programmers has inherited the same directive: don't write
it, reuse it. What has shifted, generation to generation, is how much of
someone else's architectural decisions must be carried along with the component
you actually needed.

For decades, that cost was total. A shared library arrived as a compiled `.so`
and a header file — an opaque artifact whose internals your compiler could
neither inspect nor optimize. Source-level languages like Python and JavaScript
opened the box, but the entire contents still had to be hauled along; a
`node_modules` directory stands as a monument to that reality. Bundlers
introduced selective elimination through tree-shaking, though only for code that
went entirely unmentioned by name. Rust and Go pushed specialization further into
the language semantics, yielding a telling characteristic of the modern era:
a Go binary carries no shared-library dependencies — not because it is statically
linked in the traditional sense, but because every capability arrived as source
and left as machine code specialized for that single program.

Fifty years of compiler engineering, and it all meets the same wall. The code
itself is treated as sacred. A compiler may delete unused portions of a
dependency and specialize its generic abstractions, but it may never restructure
them. The generality of the libraries a project depends on becomes the project's
generality too, whether that generality was ever wanted or not.

That is the wall AI is knocking down — not by making compilers smarter, but by
making faithful transcription cheap. When an algorithm can be re-expressed
inside a team's own structure in an afternoon rather than across two quarters of
pull requests, what is being reused is no longer the module. It is the idea. And
once ideas become the unit of reuse, every project can be bespoke.
