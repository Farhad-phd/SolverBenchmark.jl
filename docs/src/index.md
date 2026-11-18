# [SolverBenchmark.jl documentation](@id Home)

This package provides general tools for benchmarking solvers, focusing on a few
guidelines:
- The output of a solver's run on a suite of problems is a `DataFrame`, where each row
  is a different problem.
  - Since naming issues may arise (e.g., same problem with different number of
    variables), there must be an ID column;
- The collection of two or more solver runs (`DataFrame`s), is a
  `Dict{Symbol,DataFrame}`, where each key is a solver;

This package is developed focusing on
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) and
[JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl), but they should be
general enough to be used in other places.

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/SolverBenchmark.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.

# Parallel runs

By default, `solve_problems` and `bmark_solvers` solve the problems one after the other. Two
keyword arguments, accepted by both functions, solve them in parallel; choose at most one.
The solvers themselves always run one after the other.

## Threads

```julia
stats = bmark_solvers(solvers, problems; threads = 8)
```

The problems are solved in `min(8, Threads.nthreads())` tasks. Julia cannot add threads at
runtime, so start it with `julia -t 8`; a warning is printed otherwise. The solvers and the
keyword arguments passed to them are shared by all tasks, so they must not hold shared mutable
state, such as a preallocated solver workspace or a counter.

Models that wrap a native library through a pointer (CUTEst, AmplNLReader) cannot be used from
several threads and are refused with an `ArgumentError`. Use worker processes for them.

## Worker processes

```julia
using Distributed   # loads the extension of SolverBenchmark that implements `distributed`
stats = bmark_solvers(solvers, problems; distributed = 8)
```

SolverBenchmark starts 8 worker processes, solves the problems on them, and stops them when
done. Workers that existed before are not touched. Starting workers takes several seconds, so
`bmark_solvers` starts them once for all solvers. To reuse workers across calls, pass a
`WorkerPool` instead of an integer; you then load SolverBenchmark and the packages your
solvers need on its workers yourself.

Each new worker runs with one thread and one BLAS thread, uses the active project, and loads
SolverBenchmark. It then loads every package you loaded with `using`, and every submodule you
loaded with `using` directly below a package, such as `using OptimizationProblems.ADNLPProblems`.
Finally it runs `worker_setup` if given.

Anonymous functions, like `nlp -> trunk(nlp, max_time = 60.0)`, and functions from these
packages reach the workers automatically. Functions defined by name in your script do not: put
them in a file and pass `worker_setup = "my_definitions.jl"`, or pass an expression,
`worker_setup = quote ... end`. Modules loaded in another way, for instance with `import` or
deeper than one level below a package, can be loaded the same way, e.g.
`worker_setup = :(using OptimizationProblems.ADNLPProblems)`.

When `problems` is a generator, each worker builds its own models with the generator's
function, and only the generator's arguments (problem names, sizes, ...) are sent. Otherwise
the models are built in the main process and sent to the workers, which is refused for models
that wrap a native library.

If a worker dies while solving a problem, for instance after a crash in a native library, the
problem is recorded with status `:exception` and an `extrainfo` starting with
`"worker died"`, and the other workers solve the remaining problems.

With `distributed`, a `solver_logger` other than `NullLogger` is replaced on the workers by
`ConsoleLogger(stderr, level)` with the same minimum level.

## Timings

A parallel run returns the same rows, in the same order, as a sequential run, except for the
timings. Simultaneous solves compete for memory bandwidth and caches, so `elapsed_time` is
distorted and should not be compared with sequential timings. Costs based on counters
(`neval_obj`, `neval_grad`, ...) are unaffected. For the same reason, a `max_time` limit can
trigger on different problems than in a sequential run. Prefer sequential runs for
time-based performance profiles.

## CUTEst

`CUTEstModel(p)` decodes and compiles problem `p` at each construction, in a folder shared by
all processes and with file names that depend only on the problem name and precision. Decode
and compile every problem once, sequentially, then build the models with `decode = false`,
which only reads the compiled files. On Windows, this is also required because a library loaded
by one process cannot be overwritten by another.

```julia
using Distributed, CUTEst, JSOSolvers, SolverBenchmark

problem_names = CUTEst.select_sif_problems(max_con = 0, min_var = 50, max_var = 80)
# one size per SIF name: the sizes of a problem share one compiled library
@assert allunique(problem_names)
for p in problem_names  # decode and compile once, sequentially, before the benchmark
  sifdecoder(p)
  build_libsif(p)
end
problems = (CUTEstModel(p; decode = false) for p in problem_names)  # only reads files
solvers = Dict(
  :trunk => nlp -> trunk(nlp, max_time = 60.0),
  :lbfgs => nlp -> lbfgs(nlp, max_time = 60.0),
)
stats = bmark_solvers(solvers, problems; distributed = 4)
```

The problems must be given as a generator, so that each worker builds its own models.
CUTEst models are refused with `threads`.

# Tutorials

You can access more tutorials regarding [SolverBenchmark.jl](https://github.com/JuliaSmoothOptimizers/SolverBenchmark.jl) on the JuliaSmoothOptimizers website [jso.dev](https://jso.dev), for instance, an [Introduction to SolverBenchmark](https://jso.dev/tutorials/introduction-to-solverbenchmark/).
