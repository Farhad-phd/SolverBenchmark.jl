export solve_problems

"""
    solve_problems(solver, solver_name, problems; kwargs...)

Apply a solver to a set of problems.

#### Arguments
* `solver`: the function name of a solver;
* `solver_name`: name of the solver;
* `problems`: the set of problems to pass to the solver, as an iterable of
  `AbstractNLPModel`. It is recommended to use a generator expression (necessary for
  CUTEst problems).

#### Keyword arguments
* `solver_logger::AbstractLogger`: logger wrapping the solver call (default: `NullLogger`);
* `reset_problem::Bool`: reset the problem's counters before solving (default: `true`);
* `skipif::Function`: function to be applied to a problem and return whether to skip it
  (default: `x->false`);
* `colstats::Vector{Symbol}`: summary statistics for the logger to output during the
benchmark (default: `[:name, :nvar, :ncon, :status, :elapsed_time, :objective, :dual_feas, :primal_feas]`), solver's `solver_specific` scalar entries are appended as extra columns;
* `info_hdr_override::Dict{Symbol,String}`: header overrides for the summary statistics
  (default: use default headers);
* `prune`: do not include skipped problems in the final statistics (default: `true`);
* `threads::Integer`: solve the problems in this many tasks on the threads Julia was started
  with; `0` or `1` solve them one after the other (default: `0`);
* `distributed`: solve the problems on worker processes; an integer `n` starts `n` workers for
  this call and stops them afterwards, an `AbstractWorkerPool` uses its workers as they are;
  `0` solves the problems in this process (default: `0`). Requires `using Distributed`;
* `worker_setup`: code to run on each worker started by `distributed = n`, as an `Expr`
  evaluated in `Main` or the path of a file included in `Main` (default: `nothing`);
* any other keyword argument to be passed to the solver. Keywords named `threads`,
  `distributed` and `worker_setup` are not forwarded to the solver; to pass a solver option
  with one of these names, wrap the solver, e.g. `nlp -> solver(nlp; threads = 4)`.

#### Return value
* a `DataFrame` where each row is a problem, minus the skipped ones if `prune` is true.

#### Parallel runs
Choose at most one of `threads` and `distributed`. Without them, the problems are solved one
after the other. The rows of a parallel run are in the same order and hold the same values as
in a sequential run, except for timings (see below).

* `threads = n` uses `min(n, Threads.nthreads())` tasks. Julia cannot add threads at runtime:
  start it with `julia -t n` (a warning is printed otherwise). The solver and the keyword
  arguments passed to it are shared by all tasks, so they must not hold shared mutable state
  (a preallocated solver workspace, a counter, …). Models that wrap a native library through a
  pointer (CUTEst, AmplNLReader) are refused: use `distributed` for them.
* `distributed` requires `using Distributed`, which loads the extension that implements it.
  Starting workers takes several seconds at each call: prefer `bmark_solvers`, which starts them
  once for all solvers, or pass a `WorkerPool`. Each new worker runs with one thread and one
  BLAS thread, uses the active project, loads `SolverBenchmark` and every package loaded with
  `using` in `Main`, then runs `worker_setup`. Anonymous functions and functions from packages
  are sent to the workers automatically, but functions defined by name in your script are not:
  define them through `worker_setup`. With a pool, `worker_setup` is not accepted: prepare its
  workers yourself.
* With `distributed`, if `problems` is a generator, each worker builds its models with the
  generator's function; otherwise the models are built here and sent, which is refused for
  models that wrap a native library. If a worker dies, its problem is recorded with status
  `:exception` and `extrainfo` starting with `"worker died"`, and the run continues.
* With `distributed`, a `solver_logger` other than `NullLogger` is replaced on the workers by
  `ConsoleLogger(stderr, level)` with the same minimum level.
* In parallel, `elapsed_time` is distorted by contention between simultaneous solves and is not
  comparable with sequential timings, while counter-based costs (`neval_obj`, …) are
  unaffected. For the same reason, `max_time` can trigger on different problems than in a
  sequential run.
"""
function solve_problems(
  solver,
  solver_name,
  problems;
  threads::Integer = 0,
  distributed = 0,
  worker_setup = nothing,
  kwargs...,
)
  _check_parallel_keywords(threads, distributed)
  if _is_distributed(distributed)
    return _with_distributed(distributed, worker_setup) do runner
      _solve_problems(solver, solver_name, problems, runner; kwargs...)
    end
  end
  return _solve_problems(solver, solver_name, problems, _local_runner(threads); kwargs...)
end

# Shared by `solve_problems` and `bmark_solvers`: columns and header, then the runner.
function _solve_problems(
  solver,
  solver_name::TName,
  problems,
  runner::AbstractRunner;
  solver_logger::AbstractLogger = NullLogger(),
  reset_problem::Bool = true,
  skipif::Function = _never_skip,
  colstats::Vector{Symbol} = [
    :solver_name,
    :name,
    :nvar,
    :ncon,
    :status,
    :iter,
    :elapsed_time,
    :objective,
    :dual_feas,
    :primal_feas,
  ],
  info_hdr_override::Dict{Symbol, String} = Dict{Symbol, String}(:solver_name => "Solver"),
  prune::Bool = true,
  kwargs...,
) where {TName}
  f_counters = collect(fieldnames(Counters))
  fnls_counters = collect(fieldnames(NLSCounters))[2:end] # Excludes :counters
  ncounters = length(f_counters) + length(fnls_counters)
  types = [
    TName
    Int
    String
    Int
    Int
    Int
    Symbol
    Float64
    Float64
    Int
    Float64
    Float64
    fill(Int, ncounters)
    String
  ]
  names = [
    :solver_name
    :id
    :name
    :nvar
    :ncon
    :nequ
    :status
    :objective
    :elapsed_time
    :iter
    :dual_feas
    :primal_feas
    f_counters
    fnls_counters
    :extrainfo
  ]

  col_idx = indexin(colstats, names)

  @info log_header(colstats, types[col_idx], hdr_override = info_hdr_override)

  run_one = _SolveOne(
    solver,
    solver_name,
    f_counters,
    fnls_counters,
    solver_logger,
    reset_problem,
    skipif,
    (; kwargs...),
  )
  logr = r -> (prune && r.kind === :skipped) || @info log_row(r.row[col_idx])
  results = _run(runner, run_one, logr, problems)
  return _assemble(results, names, types; prune = prune)
end
