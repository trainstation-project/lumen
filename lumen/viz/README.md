# `lumen/viz`

Looking at what a compiled function does, for a person rather than the
compiler: built on the graph (`lumen/graph`), the compiler's `Plan`
(`lumen/graph_compiler`), the profiler and the kernels' sources
(`lumen/ops`, `lumen/codegen`). A compiled function's `dump_graph(path)`
and `dump_code(directory)` call in here.

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | The public names: `collect`, `write`, `dump_code`, `program_source`, `source_of`, `type_text`. |
| `common.py` | What both share: dtype names (`DTYPES`, `type_text`), the source file and anchor of an MPS primitive kernel (`source_of`), the ops' folder (`OPS`), and the Python lines a step was traced at. |
| `page.py` | `dump_graph`'s page: `profile_steps` (each step's kernels and GPU times), the plan and traced-graph views, `collect` (the page's data: graphs, plans, kernels, their sources) and `write` (the page from `page.html`). |
| `page.html` | The page template (`__TITLE__`, `__DATA__` placeholders), its CSS and JS. |
| `code.py` | `dump_code`: a call run with its launches captured (`graph_compiler._capture_launches`), each kernel's code a file in order (Metal: a fusion's generated source or a primitive kernel's template; a TPU: the Mosaic module, and its Pallas, the template's `program(spec)`; the host: the step), headed by its step's Python lines and graph; `index.txt`, `graph.txt`, `plan.txt`, the templates' sources, and `program_source` (the program as Python, `program.py`, tracing to the same graph). |
| `__main__.py` | The CLI: `python -m lumen.viz FILE.py:FUNCTION --input f32[64,128] ...` (or `--example`) writes the page. |

## Related

* [`lumen/graph`](../graph/README.md) — the graph and the tracer whose `compile` calls `dump_graph` and `dump_code`.
* [`lumen/ops/launched.rs`](../ops/launched.rs) — the launch capture `dump_code` reads.
