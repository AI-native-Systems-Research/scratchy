//! `Uniform.td` — ONE PROGRAM WRITTEN ONCE AND MAPPED ONTO MANY UNITS.
//!
//! Not one of `dcc`'s own dialects: it belongs to the dataflow scheduler
//! (`dataflow-scheduler/external/dataflow-scheduler-dialects/include/dataflow-scheduler/Dialect/Uniform/Uniform.td`,
//! dialect `uniform` at `:34-41`) and its ops arrive in the DataflowIR the scheduler hands to `dcc`.
//!
//! # ⛔⛔ THE POINT OF THE DIALECT IS THAT THE PROGRAM IS **SHARED**
//!
//! `uniform.uniformize_regions`'s own summary is *"to map parallel regions to different units, aiming
//! to consolidate programs from multiple units into a single program"* (`Uniform.td:83`). A region
//! carries a LIST of units and one region argument standing for *"whichever of them is running this"*,
//! and `uniform.query_map` is how a body reads the one constant that differs per unit. So an op count
//! is not a work count: sixteen units running one region is one region in the IR.
//!
//! ⭐ IT IS THE INPUT SHAPE `FlatteningLocalRegions` REWRITES, which is why it lands here now.
//! Entries 180, 181 and 182 ask an operation WHICH op it is —
//! `isa<uniform::UniformizeRegionsOp>` (`dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:137`,
//! `:235`) and `isa<mlir::uniform::YieldOp>` (`:229`) — and those three questions cannot be asked of
//! an island that has no `uniform` op at all. [`super::Op`]'s eighth variant is that answer.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::print;

/// ONE REGION OF A `uniform.uniformize_regions`, WITH THE UNITS IT IS MAPPED ONTO.
///
/// ```mlir
/// (%arg1 -> %0, %2){
///   ..
///   uniform.yield
/// }
/// ```
/// (`dcc/test/Transform/FlatteningLocalRegions/flatten_local_region.mlir:87`.)
///
/// # ⛔⛔ THE REFERENCE HOLDS THIS AS A FLAT OPERAND LIST PLUS A LENGTH ARRAY, AND ASSERTS THE ZIP
///
/// `UniformizeRegionsOp`'s arguments are `Variadic<AnyType>:$units` and `I32ArrayAttr:$list_sizes`
/// (`Uniform.td:86-87`), with the regions a separate `VariadicRegion` (`:91`) — three lists that must
/// agree, and the op's verifier is three `assert`s saying so:
///
/// ```cpp
/// assert(op.getRegions().size() >= 1);
/// assert(op.getRegions().size() == op.getListSizes().size());
/// // ..
/// assert((accu_size == size) &&
///        "unit size has to be matched to the list sizes contents.");
/// ```
/// (`dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Uniform/Uniform.cpp:125-136`.)
///
/// ⭐ ZIPPING THEM INTO ONE RECORD PER REGION MAKES ALL THREE UNWRITABLE. That is not an invention
/// either — the reference reaches for the zipped form every time it wants to *use* the op:
/// `getRegionUnitList(int pos)` re-slices `$units` by the prefix sum of `$list_sizes` on every call
/// (`Uniform.cpp:184-192`), and `getUnitsPerRegionsAsVectorOfVector` exists for no other purpose than
/// to materialise the whole vector-of-vectors (`dcc/src/Dialect/Uniform/Utils.cpp:490-504`), which is
/// what both `FlatteningLocalRegionsTree::compute` (`FlatteningLocalRegions.cpp:157-158`) and
/// `flatten` (`:389-390`) call before they do anything.
///
/// ⛔ A REGION WITH NO UNITS IS STILL EXPRESSIBLE and stays so: `list_sizes` may hold a 0 and the
/// verifier's prefix sum still balances. `getRegionUnitList` would return an empty range, and
/// `traverseRegion` (entry 180) would attribute the region's operations to nobody.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRegion {
    /// `getRegionArg(i)` — *"whichever unit of [`LocalRegion::units`] is running this region"*.
    ///
    /// ⛔⛔ THE REGION'S ONE BLOCK ARGUMENT, AND IT IS BOUND BY THE REGION AND NOT BY THE OP.
    /// `Value getRegionArg(int i) {return getRegion(i).getArgument(0);}` (`Uniform.td:96`) — so each
    /// region binds its own, and two regions of one op have two different arguments. The parser
    /// enforces the arity: it parses exactly one `OpAsmParser::Argument` per region and passes it to
    /// `parseRegion` (`Uniform.cpp:78-81`), and its type is always `index` (`:68-69`).
    ///
    /// ⭐ ENTRY 182 REMAPS EXACTLY THIS VALUE. `cloneOpsForRegions` finds the region of the parent
    /// `uniformize_regions` that holds the operation it is cloning and maps that region's argument to
    /// the NEW op's block argument (`FlatteningLocalRegions.cpp:246-256`) — which is why the vendor's
    /// `uniform.query_map(map:%285, key:%arg48)` comes out as
    /// `uniform.query_map(map:%VAL_356, key:%VAL_351)`
    /// (`flatten_local_region4.mlir:751` against `:355`). Without the argument as a field there is
    /// nothing to be the source of that mapping.
    pub arg: Val,
    /// The units this region runs on — `getRegionUnitList(i)` (`Uniform.cpp:184`).
    ///
    /// ⭐ ORDER IS OBSERVABLE, TWICE OVER. It is the order the printer writes them in
    /// (`p.printOperands(op.getRegionUnitList(i))`, `Uniform.cpp:110`), and it is the order
    /// `traverseRegion` pushes them into `unit_to_ops` — whose FIRST-INSERTION key order is this
    /// pass's output region order (see
    /// [`crate::bridges::dataflow_ir_to_sentient::tf_flattening_local_regions::UnitToOps`]).
    pub units: Vec<Val>,
    /// What the region runs, ending in a [`Op::Yield`].
    ///
    /// ⭐ THE TERMINATOR IS PART OF THE LIST, NOT IMPLIED BY IT. `ImplicitUniformTerminator`
    /// (`Uniform.td:60-61`, `:80`) means the PARSER inserts one when the text omits it
    /// (`ensureTerminator`, `Uniform.cpp:86`) — but the printer prints it
    /// (`printRegion(.., /*printBlockTerminators=*/true)`, `:112`) and, decisively, entry 182 has to
    /// SEE it: `cloneOpsForRegions` skips a `uniform.yield` node explicitly
    /// (`FlatteningLocalRegions.cpp:229-232`) and `flatten` appends a fresh one to every region it
    /// builds (`:450-451`). A representation that implied the terminator would make both of those
    /// unwritable.
    pub body: Vec<super::Op>,
}

/// ONE `uniform` OPERATION.
///
/// ⚠️ FOUR OF `Uniform.td`'S FIVE. `uniform.equalize_pattern` (`:187`) is declared beside them and is
/// absent: no bridge-2 function this campaign has reached names it, and an op nothing reads would be
/// a variant every total match in this island has to answer for with nothing to say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `uniform.uniformize_regions -> () { (%arg -> %0, %2){ .. } .. }` — THE DIALECT'S MAIN OP.
    ///
    /// # ⭐ WHAT `FlatteningLocalRegions` EXISTS TO REWRITE
    ///
    /// The pass rebuilds one of these so that its regions are the EQUIVALENCE CLASSES of its units
    /// rather than the nesting the scheduler happened to emit — its own worked example is
    /// `FlatteningLocalRegions.cpp:316-381`, and it declines outright when the class count already
    /// equals the region count (`:418`).
    ///
    /// ⛔⛔ THE NESTING IS THE INPUT, NOT AN ARTEFACT. A region of one of these holds another one
    /// (`flatten_local_region.mlir:86-88`, three deep), and the inner op's regions split the outer
    /// op's unit list further. That is why the pass needs a tree at all.
    UniformizeRegions {
        /// One record per region, in region order — see [`LocalRegion`].
        regions: Vec<LocalRegion>,
        /// `Variadic<AnyType>:$results` (`Uniform.td:90`) — what the op binds, one per operand of
        /// every region's [`Op::Yield`].
        ///
        /// ⛔ NOT ALWAYS EMPTY IN THE INPUT: `%69 = uniform.uniformize_regions -> index {`
        /// (`dcc/test/Conversion/SentientToProgIR/uniformization.mlir:789`) and
        /// `%[[VAL_171]]:2 = uniform.uniformize_regions -> (index, index) {`
        /// (`dcc/test/Transform/LiveRangeReduction/uniformizeRegions.mlir:174`) — 46 and 1
        /// occurrences under `dcc/test` against 305 result-less ones. The op's verifier ties the
        /// count to every region's terminator (`Uniform.cpp:170-179`), so dropping the field would
        /// make that relation unstatable.
        ///
        /// ⭐ AND THE ONE OP THIS CRATE **BUILDS** HAS NONE: `flatten` creates its replacement with
        /// `mlir::TypeRange()` (`FlatteningLocalRegions.cpp:421`), so the emitted form is always
        /// `-> ()`.
        results: Vec<Val>,
    },

    /// `uniform.yield` — the implicit terminator of every region of the two ops above.
    ///
    /// ⛔⛔ ENTRY 182 TESTS FOR IT BY NAME AND SKIPS IT. `if (isa<mlir::uniform::YieldOp>(..))
    /// { node = node->getNextSibling(); continue; }` (`FlatteningLocalRegions.cpp:229-232`) — the old
    /// terminators are never cloned, because `flatten` appends a fresh one per new region
    /// (`:450-451`). An island in which a terminator were indistinguishable from any other op would
    /// clone them and emit a region with two terminators.
    Yield {
        /// `Variadic<AnyType>:$operands` (`Uniform.td:71`) — usually none.
        ///
        /// ⭐ THE COUNT IS THE PARENT'S RESULT COUNT, and the parent's verifier says so
        /// (`Uniform.cpp:170-179`). The vendor writes both forms: a bare `uniform.yield`
        /// (`flatten_local_region.mlir:103`) and `uniform.yield %32 : index`.
        operands: Vec<Val>,
    },

    /// `%m = uniform.def_immutable_mapping([%2 -> %65], [%6 -> %67], ..):index` — THE PER-UNIT
    /// CONSTANTS OF ONE OTHERWISE-SHARED PROGRAM.
    ///
    /// > *"Constructs an immutable key-value mapping from one set of SSA values to another and returns
    /// > a handle to it. It is used to capture **constant** differences between the per-core programs
    /// > of a unit (for example differing loop bounds or addresses): each core's value is associated
    /// > with a common key, so the program can be expressed once and specialized per core via
    /// > `uniform.query_map`."*
    ///
    /// (`Uniform.td:118-125`.)
    ///
    /// ⭐ PRESENT BECAUSE IT IS IN ENTRY 182'S INPUT. The authority's only case that reaches
    /// `cloneOpsForRegions` clones one of these (`flatten_local_region4.mlir:750` into `:354`), and a
    /// stand-in op would make the test a test of a different program.
    DefImmutableMapping {
        /// The `index` handle the op binds — `Index:$result` (`Uniform.td:134`).
        result: Val,
        /// `$keys` zipped with `$values`, *"paired positionally"* (`Uniform.td:124-125`).
        ///
        /// ⭐ ONE LIST, NOT TWO, for the same reason [`LocalRegion`] is one record: the two operand
        /// ranges are `AttrSizedOperandSegments` (`Uniform.td:116`) of necessarily equal length, and
        /// the printer walks them by a single index (`Uniform.cpp:455-465`). The keys are the units —
        /// the verifier checks they are all `dataflow.get_unit`s when the first one is
        /// (`Uniform.cpp:477-486`).
        pairs: Vec<(Val, Val)>,
    },

    /// `%v = uniform.query_map(map:%m, key:%arg0) : index` — READ ONE UNIT'S CONSTANT OUT OF A
    /// MAPPING.
    ///
    /// ⭐ THE KEY IS ORDINARILY THE ENCLOSING REGION'S ARGUMENT, which is what makes it the op that
    /// specialises a shared program: `uniform.query_map(map:%285, key:%arg48)`
    /// (`flatten_local_region4.mlir:751`), with `%arg48` the [`LocalRegion::arg`] of the region it
    /// sits in. ⛔ AND THAT IS WHY ENTRY 182'S BLOCK-ARGUMENT REMAPPING IS OBSERVABLE AT ALL — the
    /// clone reads the NEW region's argument (`:355`).
    QueryMap {
        /// The value it binds — `AnyType:$result` (`Uniform.td:177`), `index` in this island.
        result: Val,
        /// `Index:$map` — the handle a [`Op::DefImmutableMapping`] bound.
        map: Val,
        /// `Index:$key` — which unit's value to read.
        key: Val,
    },
}

/// ONE `uniform` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::UniformizeRegions { regions, results } => {
            // ⛔ `printArrowTypeList` IS NOT OPTIONAL AND PARENTHESISES ALL BUT ONE. The printer's
            // first act is `p.printArrowTypeList(op.getResultTypes())` (`Uniform.cpp:103`), and
            // MLIR's own rule is bare for a single non-function type and parenthesised otherwise —
            // which is exactly the three forms the vendor's files contain: `-> ()` (305 times),
            // `-> index` (46) and `-> (index, index)` (1).
            let result_tys = match results.len() {
                0 => " -> ()".to_string(),
                1 => " -> index".to_string(),
                n => format!(" -> ({})", vec!["index"; n].join(", ")),
            };
            // ⚠️ MLIR NUMBERS AN UNNAMED MULTI-RESULT OP AS `%171:2 = `, and this island writes the
            // comma-separated form its `scf.if` already writes — see
            // [`super::scf::Op::If::results`]. Nothing here emits one: `flatten` builds its
            // replacement with `mlir::TypeRange()` (`FlatteningLocalRegions.cpp:421`).
            let bound = if results.is_empty() {
                String::new()
            } else {
                format!("{} = ", print::vals(results))
            };
            let _ = writeln!(out, "{bound}uniform.uniformize_regions{result_tys} {{");
            for region in regions {
                print::indent(out, depth + 1);
                // ⛔ NO SPACE BEFORE THE BRACE. `p << '(' << arg << " -> "; printOperands(units);
                // p << ')'; p.printRegion(..)` (`Uniform.cpp:109-112`) puts the region's own `{`
                // straight after the `)`, and the vendor's text is `(%arg1 -> %0, %2){`
                // (`flatten_local_region.mlir:87`) — one character apart from every other region
                // header in this island.
                let _ = writeln!(
                    out,
                    "({} -> {}){{",
                    print::val(region.arg),
                    print::vals(&region.units)
                );
                // ⭐ THE TERMINATOR PRINTS. `printRegion(region, /*printEntryBlockArgs=*/false,
                // /*printBlockTerminators=*/true)` (`:112`) — the argument is already in the header
                // above, the `uniform.yield` is not.
                for inner in &region.body {
                    print::emit(out, inner, depth + 2);
                }
                print::indent(out, depth + 1);
                out.push_str("}\n");
            }
            print::indent(out, depth);
            out.push_str("}\n");
        }
        Op::Yield { operands } => {
            if operands.is_empty() {
                out.push_str("uniform.yield\n");
            } else {
                // `attr-dict ($operands^ `:` type($operands))?` (`Uniform.td:76`) — the type list is
                // not optional once there are operands, exactly as for `scf.yield`.
                let _ = writeln!(
                    out,
                    "uniform.yield {} : {}",
                    print::vals(operands),
                    vec!["index"; operands.len()].join(", ")
                );
            }
        }
        Op::DefImmutableMapping { result, pairs } => {
            // ⛔ `"):"` WITH NO SPACES AROUND THE COLON, and the trailing value type elided when it
            // equals the result type (`Uniform.cpp:456-472`). Every value in this island is `index`,
            // so the `, <value type>` tail at `:469-472` never prints — and the vendor's own line is
            // `..[%62 -> %95]):index` (`flatten_local_region4.mlir:750`).
            let _ = writeln!(
                out,
                "{} = uniform.def_immutable_mapping({}):index",
                print::val(*result),
                pairs
                    .iter()
                    .map(|(key, value)| format!(
                        "[{} -> {}]",
                        print::val(*key),
                        print::val(*value)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Op::QueryMap { result, map, key } => {
            // `p << "(map:"; .. p << ", key:"; .. p << ") : "` (`Uniform.cpp:586-591`) — no space
            // after either colon, and one on each side of the result type's.
            let _ = writeln!(
                out,
                "{} = uniform.query_map(map:{}, key:{}) : index",
                print::val(*result),
                print::val(*map),
                print::val(*key)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::uniform::{LocalRegion, Op};
    use crate::islands::dataflow_ir::dialects::{self, Val, arith};
    use crate::islands::dataflow_ir::print::emit;

    /// ⭐⭐ IBM'S OWN NESTED REGION, REPRODUCED BYTE FOR BYTE.
    ///
    /// `dcc/test/Transform/FlatteningLocalRegions/flatten_local_region.mlir:86-88` and its two closing
    /// braces at `:117-119` — the input shape `FlatteningLocalRegions` is given, with the inner
    /// region's body elided to the terminator so that the test is about the op's own syntax.
    #[test]
    fn a_uniformize_regions_prints_as_the_reference_writes_it() {
        let inner = dialects::Op::Uniform(Op::UniformizeRegions {
            regions: vec![LocalRegion {
                arg: Val(2),
                units: vec![Val(0)],
                body: vec![dialects::Op::Uniform(Op::Yield {
                    operands: Vec::new(),
                })],
            }],
            results: Vec::new(),
        });
        let outer = dialects::Op::Uniform(Op::UniformizeRegions {
            regions: vec![LocalRegion {
                arg: Val(1),
                units: vec![Val(0), Val(2)],
                body: vec![
                    inner,
                    dialects::Op::Uniform(Op::Yield {
                        operands: Vec::new(),
                    }),
                ],
            }],
            results: Vec::new(),
        });

        let mut got = String::new();
        emit(&mut got, &outer, 0);

        assert_eq!(
            got,
            "uniform.uniformize_regions -> () {\n  \
             (%1 -> %0, %2){\n    \
             uniform.uniformize_regions -> () {\n      \
             (%2 -> %0){\n        \
             uniform.yield\n      \
             }\n    \
             }\n    \
             uniform.yield\n  \
             }\n\
             }\n"
        );
    }

    /// AND A RESULT-BINDING ONE PRINTS ITS TYPE BARE.
    ///
    /// `%69 = uniform.uniformize_regions -> index {`
    /// (`dcc/test/Conversion/SentientToProgIR/uniformization.mlir:789`) — one result, no parentheses,
    /// and the region's terminator carries the operand the result comes from.
    #[test]
    fn a_result_binding_uniformize_regions_prints_a_bare_type() {
        let op = dialects::Op::Uniform(Op::UniformizeRegions {
            regions: vec![LocalRegion {
                arg: Val(70),
                units: vec![Val(1)],
                body: vec![dialects::Op::Uniform(Op::Yield {
                    operands: vec![Val(71)],
                })],
            }],
            results: vec![Val(69)],
        });

        let mut got = String::new();
        emit(&mut got, &op, 0);

        assert_eq!(
            got,
            "%69 = uniform.uniformize_regions -> index {\n  \
             (%70 -> %1){\n    \
             uniform.yield %71 : index\n  \
             }\n\
             }\n"
        );
    }

    /// ⭐⭐ THE MAPPING PAIR AND ITS QUERY, AS THE VENDOR'S OWN CASE WRITES THEM.
    ///
    /// `flatten_local_region4.mlir:750-752`, shortened to the first two of its sixteen pairs — the
    /// three operations entry 182 clones in the authority's only case that reaches it.
    #[test]
    fn the_mapping_ops_print_as_the_reference_writes_them() {
        let ops = vec![
            dialects::Op::Uniform(Op::DefImmutableMapping {
                result: Val(285),
                pairs: vec![(Val(2), Val(65)), (Val(6), Val(67))],
            }),
            dialects::Op::Uniform(Op::QueryMap {
                result: Val(286),
                map: Val(285),
                key: Val(48),
            }),
        ];

        let mut got = String::new();
        for op in &ops {
            emit(&mut got, op, 0);
        }

        assert_eq!(
            got,
            "%285 = uniform.def_immutable_mapping([%2 -> %65], [%6 -> %67]):index\n\
             %286 = uniform.query_map(map:%285, key:%48) : index\n"
        );
    }

    /// THE WALKS AGREE WITH THE OP'S OWN THREE LISTS.
    ///
    /// ⛔⛔ THE UNITS ARE OPERANDS AND THE REGION ARGUMENTS ARE NOT. `$units` is an operand range
    /// (`Uniform.td:86`) while `getRegionArg(i)` is a BLOCK argument (`:96`), and a census that
    /// confused the two would report a value as both read and bound — which is how a clone
    /// re-defines a name. [`dialects::operands`] must flatten the per-region lists back into the
    /// reference's single `$units` range, in region order, because that is the order the verifier's
    /// prefix sum walks (`Uniform.cpp:184-192`).
    #[test]
    fn the_units_are_operands_and_the_region_arguments_are_bound() {
        let op = dialects::Op::Uniform(Op::UniformizeRegions {
            regions: vec![
                LocalRegion {
                    arg: Val(100),
                    units: vec![Val(0), Val(2)],
                    body: vec![dialects::Op::Arith(arith::Op::Constant {
                        result: Val(9),
                        value: 0,
                    })],
                },
                LocalRegion {
                    arg: Val(101),
                    units: vec![Val(1)],
                    body: Vec::new(),
                },
            ],
            results: vec![Val(102)],
        });

        assert_eq!(dialects::operands(&op), vec![Val(0), Val(2), Val(1)]);
        assert_eq!(dialects::block_args(&op), vec![Val(100), Val(101)]);
        assert_eq!(dialects::results(&op), vec![Val(102)]);
        assert_eq!(dialects::regions(&op).len(), 2, "one per `LocalRegion`");
        assert_eq!(dialects::regions(&op)[1], &[] as &[dialects::Op]);
    }

    /// AND A QUERY READS BOTH OF ITS OPERANDS, WHICH IS WHAT MAKES THE CLONE REWRITE BOTH.
    ///
    /// Entry 182 rewrites a clone's operands through the mapping it was given, so an operand the
    /// census misses is an operand that keeps pointing at the ORIGINAL region's argument.
    #[test]
    fn a_query_reads_its_map_and_its_key() {
        let op = dialects::Op::Uniform(Op::QueryMap {
            result: Val(286),
            map: Val(285),
            key: Val(48),
        });
        assert_eq!(dialects::operands(&op), vec![Val(285), Val(48)]);
        assert_eq!(dialects::results(&op), vec![Val(286)]);

        let mapping = dialects::Op::Uniform(Op::DefImmutableMapping {
            result: Val(285),
            pairs: vec![(Val(2), Val(65))],
        });
        // ⭐ KEYS AND VALUES BOTH, KEY FIRST — the two `Variadic` ranges in declaration order
        // (`Uniform.td:132-133`).
        assert_eq!(dialects::operands(&mapping), vec![Val(2), Val(65)]);

        // ⛔ A TERMINATOR READS WHAT IT YIELDS AND DEFINES NOTHING.
        let yielded = dialects::Op::Uniform(Op::Yield {
            operands: vec![Val(71)],
        });
        assert_eq!(dialects::operands(&yielded), vec![Val(71)]);
        assert!(dialects::results(&yielded).is_empty());
    }
}
