# Systems Must Evolve to be Compilers

Here is the hot take: AI coding agents will eliminate the software
system as a concept. In its place every solution will be bespoke, span
the full stack, and will be generated from scratch. To understand why,
consider three options as we choose how to employ our new AI toy.
 
1. Have AI generate 1M lines of vLLM-esque code, and hope this code is
correct and maintainable.
2. Generate specifications, hope they are correct, and then and have
AI turn these into 1M lines of code that we hope is both maintainable
and faithful to the spec.
3. Have AI generate 50k lines of code that compiles domain-specific
languages down to a family of hyper-specialized binaries, images, boot
loaders, etc.

Our bet is that Option 3 will be more cost effective, increase our
delivery velocity, and produce smaller assets that boot and run more
quickly. With AI, all systems will evolve to become compilers.

Of course this only works if we have a reliable way to get AI
generating those 50k lines.

We further argue that by using Rust procedural macros, const generics,
and conditional compilation, all systems become hyper-specializing
meta-compilers. AI writes only the proc macros, which in turn
macro-expand DSLs to a peculiar “dialect” of strongly typed Rust. The
underlying rustc compiler does the actual heavy lifting, without us
having to touch LLVM. By restricting AI to write proc macros that
generate a subset of Rust, we protect ourselves from the AI
foot-gunning of which I am sure we are all painfully aware. The Rust
type system, over this subset, enables turning most classes of runtime
errors into compile time invariant violations — not fails to run and
also not doesn’t compile, but something far stronger: violates a
typestate invariant.
