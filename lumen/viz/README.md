# viz

Tools for looking at what the compiler did: an HTML page of a traced graph next to its fused plan with kernel timings, and a dump of every kernel's generated code. It reads graphs and plans; it never changes them. `Compiled.dump_graph` and `Compiled.dump_code` (`lumen/graph/tracing/tracer.py`) call into it.

## Files

- `__init__.py` exports `collect`, `write`, `dump_code`, `program_source`, `source_of` and `type_text`.
- `page.py` builds the page data and writes it. `collect` gathers the traced graph, the fused plan and an unfused plan of the same graph, profiling both on the device (`profile_steps`) except on CPU and meta; `graph_view` and `plan_view` turn them into nodes and edges; `code_of` pulls the traced Python source; `write` fills the template, optionally also writing the data as JSON, or a fragment without the doctype for a host page that wraps it.
- `page.html` is the self-contained template (inline CSS and script). `write` replaces `__TITLE__` and `__DATA__`.
- `code.py` holds `dump_code`, which runs a compiled program while capturing its launches (`lumen/profiler/launched.rs`) and writes each launch's kernel source to its own file in launch order, with a header naming the step, its graph and where it was traced, plus the graph, the plan and an index. `program_source` turns a graph back into Python that re-traces it through `prims.bind`. Pallas launches get a script that rebuilds their Mosaic program from its template and spec.
- `common.py` holds `type_text` (compact types such as `f32[64,128]`), `source_of` (file and anchor of a primitive kernel's source), the dtype names and the Metal prelude generated kernels are compiled with.
- `__main__.py` is a command line tool (`python -m lumen.viz`) that compiles a function (`file.py:fn`, `module:fn` or a built-in example) on inputs given as type strings and writes its page.

## Conventions

- JSON embedded in the page escapes `</` so the data cannot close the script tag.
- Unfused plans are built with `Plan(graph, device, fuse=False)`, so the page compares the fused plan with the same graph run one primitive per step.
- Kernel source lookup (`source_of`) depends on anchors in the codegen sources; renaming a kernel there can break the code view.
