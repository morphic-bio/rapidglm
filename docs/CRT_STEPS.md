# The steps of the CRT null draws

At high MOI nearly all the time of a perturbation screen goes to the null draws
of the conditional randomization test (CRT): for each pair, hundreds to
hundreds of thousands of random treatment assignments are generated and each is
scored against the response's null fit. Three steps made these draws faster
during development. `benchmarks/crt_steps.cpp` and
`scripts/benchmark_crt_steps.py` time each step with the library as released:
the benchmark calls RapidGLM and changes none of it, so the numbers belong to
the release that is built.

## What is compared

| Configuration | Executable | Mode | What it adds |
| --- | --- | --- | --- |
| `first` | `rapidglm_crt_steps_portable` | `first` | The first implementation: one draw generated and then scored, on its own |
| `single` | `rapidglm_crt_steps_portable` | `single` | The released sampler, generator and score records, still one draw at a time |
| `batched` | `rapidglm_crt_steps_portable` | `batched` | **Step 1.** A pair's draws generated 256 at a time and scored together in blocks of cells |
| `pipeline` | `rapidglm_crt_steps_portable` | `pipeline` | **Step 2.** All threads on a few pairs of one response at a time; one sampler per target |
| `vector` | `rapidglm_crt_steps` | `pipeline` | **Step 3.** The generator's Philox blocks eight at a time with AVX2 |

Each configuration changes one thing from the one above it.

- **`first`** is the draw code of the commit before the three steps (the parent
  of `e3a2da3`), carried in the benchmark file in `namespace first`: a thread
  per response with its pairs as tasks of that thread's team, a sampler built
  for every pair, 16 draws to a task, random numbers one Philox block at a time,
  and a cell's score terms read from three arrays. The released library no
  longer holds this code, so the benchmark does. It draws the same random
  numbers and adds the same sums in the same order as the released code.
- **`single`** keeps that schedule and draws one at a time, but with the
  released sampler (`crt_sampler`), generator (`CounterRng`) and packed score
  records (`BinaryScoreWorkspace`). It separates what step 1 gained from the
  released per-draw code from what it gained from batching.
- **`batched`** generates and scores with `detail::BinaryDrawBatch`, 256 draws
  to a task, on the same schedule.
- **`pipeline`** runs `detail::pair_pipeline`, the ring of active pairs, and
  builds a target's sampler once for all its pairs.
- **`vector`** is the same mode in the native build. The portable executable is
  compiled with `RAPIDGLM_FORCE_PORTABLE`. In the timed part the two differ in
  the generator (eight Philox blocks at a time in both; scalar code against
  AVX2) and, where a sampler is built, in the exponentials of its
  probabilities. The scoring and the thinned sampler are the same code.

Step 1 is `first` to `batched`, step 2 `batched` to `pipeline`, step 3
`pipeline` to `vector`.

`vector` is what `rapidglm sceptre` runs with `crt_draw: thinned` and
`batch_scoring: false`. Two later changes are the defaults from 1.6.0 and are
settings of `rapidglm sceptre`, so they are timed with the program itself:
batch scoring (`batch_scoring`) and the hazard sampler (`crt_draw: hazard`);
see [CLI.md](CLI.md). With `--hazard` the driver also times the native pipeline
with the hazard sampler. That sampler draws different random numbers, so its
table is compared only with `sceptre()`.

## What is timed

Only the null draws. Reading the inputs, cell and pair QC, the null fits, the
propensity fits, the score records and the observed statistics are done first,
as `sceptre()` does them, a chunk of responses at a time (`chunk_responses`,
256 by default), and are reported as `seconds.prepare`. The draws of each chunk
are timed with a steady clock and summed (`seconds.draws`). In `first`,
`single` and `batched` a pair's sampler is built inside the timed part, once
per pair; in `pipeline`, once per target.

The settings are those of `rapidglm sceptre`: staged budgets, the screen, the
fit check and confirmation draws all apply, so the number of draws is that of a
real run. Effect fits are not made.

## Checks

- A pair's stream is keyed by its response and target, as in `sceptre()`, so
  every mode draws the same random numbers. The driver requires the tables of
  the same build and sampler to be identical byte for byte.
- With `verify` the program also runs `sceptre()` on the same inputs and
  compares every pair's status, counts, draws, statistic and p-value bit for
  bit. The driver sets it on the first run of each build.
- The two builds may round their fits differently. Between them the driver
  compares every pair: statistics and p-values to a relative tolerance
  (`--tolerance`, 1e-6), and it counts pairs that ended at another stage.
- The driver stops if the native executable did not use the AVX2 generator or
  the portable one did (a host without AVX2, or a library configured with
  `RAPIDGLM_FORCE_PORTABLE`).

## Running it

Build both executables:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRAPIDGLM_BENCHMARKS=ON
cmake --build build -j 4 --target rapidglm_crt_steps rapidglm_crt_steps_portable
```

Prepare a panel of the Gasperini screen (207,324 cells):

```bash
python3 scripts/prepare_benchmark_data.py gasperini
python3 scripts/prepare_crt_panels.py --output validation/work/crt-panels 50x1000 100x7418
```

Describe it in the keys of `rapidglm sceptre`, for example `panel.json`:

```json
{
  "responses": "validation/work/crt-panels/responses_g1000.mtx",
  "assignments": "validation/work/benchmark/gasperini/assignments.mtx",
  "design": "validation/work/benchmark/gasperini/design.tsv",
  "guides": "validation/work/benchmark/gasperini/guides.tsv",
  "pairs": "validation/work/crt-panels/pairs_t50_g1000.tsv",
  "moi": "high",
  "cell_qc": false,
  "seed": 20260930
}
```

Paths are read from the directory the driver is run in; the script's docstrings
name the public files the panels are made from. The driver sets `threads`,
`output`, `crt_draw` and `verify` itself. Then:

```bash
python3 scripts/benchmark_crt_steps.py --config panel.json --threads 16 --cpus 6-21
```

It runs one untimed pass (which also loads the inputs into the page cache) and
three timed passes, alternating the order of the configurations between passes,
with `OPENBLAS_NUM_THREADS=1`. `--cpus` runs every command under `taskset -c`
and records, for each command, the processor time that other processes used on
those processors (`other_cpu_seconds`: the processors' busy time less the
command's own, so only roughly) and on the other hardware threads of their
cores (`sibling_cpu_seconds`), which a benchmark with one thread per core
leaves idle.
`--quiet` runs each through `scripts/quiet_run.py`, which waits for a quiet
host and holds the lock named by `RAPIDGLM_TIMING_LOCK`. `--resume` keeps the
runs already recorded. For a panel whose `first` takes hours, `--no-warmup
--repetitions 1` makes one run of each.

One mode can also be run directly:

```bash
build/rapidglm_crt_steps pipeline config.json
```

where `config.json` also names `threads` and `output`. The program accepts a
high-MOI screen with complement controls, CRT, union integration and identity
stream keys; `first` needs `crt_draw: thinned` or `direct`, the samplers the
first implementation had.

## Output

Under `--work` (default `validation/work/crt-steps`):

- `<configuration>/pass<N>.config.json`, `.run.json` and, for the first run of
  each build and sampler, `.tsv`: the configuration, the run record and the
  result table. Later identical tables are deleted.
- `runs.json`: every run, with its command, start, load before it, wall time,
  `others` (above) and `seconds` (`load`, `setup`, `prepare`, `draws`,
  `verify`, `total`).
- `summary.json`: the commit, host, threads, the hashes of both executables,
  and for each configuration the draw seconds of each timed pass, their median,
  the null draws, the thread time per draw (median draw seconds × threads /
  null draws, in microseconds) and the table's hash; `gains`, the ratios of
  median draw seconds between configurations (`first_to_batched`,
  `batched_to_pipeline`, `pipeline_to_vector`, `first_to_vector` and the two
  halves of step 1); and `builds_compared`.

## Reading the numbers

- The times are of the null draws, not of whole runs. A whole run adds the
  preparation, which no step changes.
- During development each step was timed once, on whole runs of the
  development builds ([CRT_BATCHING.json](../validation/CRT_BATCHING.json)).
  Those ratios include the preparation and are not the ones measured here.
- Step 3 compares scalar code that computes eight blocks per call with the
  AVX2 code. Before the step the generator computed two blocks per call.
- The gains depend on the panel, the processor and the thread count.
