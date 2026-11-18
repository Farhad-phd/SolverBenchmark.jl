# Optional parallel execution for `solve_problems` and `bmark_solvers`.
#
# `_solve_one` solves one problem and returns its row without touching shared state. A runner
# applies it to every problem (one after the other, on threads, or on worker processes through
# the `SolverBenchmarkDistributedExt` extension), and `_assemble` builds the `DataFrame` exactly
# as the original sequential loop did.

import LinearAlgebra

abstract type AbstractRunner end

# Solve the problems one after the other in the current task.
struct SerialRunner <: AbstractRunner end

# Solve the problems in `ntasks` tasks on the threads Julia was started with.
struct ThreadedRunner <: AbstractRunner
  ntasks::Int
end

# Default `skipif`. A named function (not a closure) so that it can be sent to workers.
_never_skip(problem) = false

# Outcome of one problem. `kind` is `:solved`, `:exception` or `:skipped`; `row` holds the base
# columns (without solver_specific values); `specific` holds all solver_specific entries of a
# solved problem, in the iteration order of its `Dict`.
_result(id, kind, row, specific = Pair{Symbol, Any}[]) =
  (id = id, kind = kind, row = row, specific = specific)

# Row of a skipped problem or of a failed solve, as the original sequential loop builds it.
function _failed_row(solver_name, problem_info, ncounters, extrainfo)
  return [
    solver_name
    problem_info
    :exception
    Inf
    Inf
    0
    Inf
    Inf
    fill(0, ncounters)
    extrainfo
  ]
end

# Body of the original sequential loop for one problem. It mutates nothing shared and finalizes
# `problem` on every path. `skipif` stays outside the `try`/`catch`, so its errors propagate.
function _solve_one(
  solver,
  solver_name,
  id,
  problem,
  f_counters,
  fnls_counters;
  solver_logger::AbstractLogger = NullLogger(),
  reset_problem::Bool = true,
  skipif = _never_skip,
  kwargs...,
)
  ncounters = length(f_counters) + length(fnls_counters)
  try
    if reset_problem
      NLPModels.reset!(problem)
    end
    nequ = problem isa AbstractNLSModel ? problem.nls_meta.nequ : 0
    problem_info = [id; get_name(problem); get_nvar(problem); get_ncon(problem); nequ]
    if skipif(problem)
      return _result(id, :skipped, _failed_row(solver_name, problem_info, ncounters, "skipped"))
    end
    try
      s = with_logger(solver_logger) do
        solver(problem; kwargs...)
      end
      specific = Pair{Symbol, Any}[k => v for (k, v) in s.solver_specific]
      counters_list = [getfield(NLPModels, f)(problem) for f in f_counters]
      nls_counters_list =
        problem isa AbstractNLSModel ? [getfield(problem.counters, f) for f in fnls_counters] :
        zeros(Int, length(fnls_counters))
      row = [
        solver_name
        problem_info
        s.status
        s.objective
        s.elapsed_time
        s.iter
        s.dual_feas
        s.primal_feas
        counters_list
        nls_counters_list
        ""
      ]
      return _result(id, :solved, row, specific)
    catch e
      @error "caught exception" e
      return _result(id, :exception, _failed_row(solver_name, problem_info, ncounters, string(e)))
    end
  finally
    finalize(problem)
  end
end

# A `ConsoleLogger` holds a stream and a lock and is not meant to be sent to another process.
# Workers receive only its minimum level and build `ConsoleLogger(stderr, min_level)` themselves.
struct _WorkerLogger
  min_level::LogLevel
end

_portable_logger(logger::NullLogger) = logger
_portable_logger(logger::AbstractLogger) = _WorkerLogger(Logging.min_enabled_level(logger))

_logger(logger::AbstractLogger) = logger
_logger(logger::_WorkerLogger) = ConsoleLogger(stderr, logger.min_level)

# Applies `_solve_one` with the options of one `solve_problems` call. A struct defined in this
# module (not a closure) so that it can be sent to workers.
struct _SolveOne
  solver
  solver_name
  f_counters::Vector{Symbol}
  fnls_counters::Vector{Symbol}
  solver_logger
  reset_problem::Bool
  skipif
  kwargs::NamedTuple
end

function (r::_SolveOne)(id, problem)
  return _solve_one(
    r.solver,
    r.solver_name,
    id,
    problem,
    r.f_counters,
    r.fnls_counters;
    solver_logger = _logger(r.solver_logger),
    reset_problem = r.reset_problem,
    skipif = r.skipif,
    r.kwargs...,
  )
end

# Copy of `r` that can be sent to workers: the logger is replaced by its minimum level.
function _for_workers(r::_SolveOne)
  return _SolveOne(
    r.solver,
    r.solver_name,
    r.f_counters,
    r.fnls_counters,
    _portable_logger(r.solver_logger),
    r.reset_problem,
    r.skipif,
    r.kwargs,
  )
end

# Job sent to a worker for each `(id, spec)`: build the model with `make(spec)`, then solve it.
struct _RemoteJob{F, R}
  make::F
  run_one::R
end

function (job::_RemoteJob)(item)
  id, spec = item
  return job.run_one(id, job.make(spec))
end

_spec_name(spec::AbstractNLPModel) = get_name(spec)
_spec_name(spec) = string(spec)

# Result of a problem whose worker died: no model is available on the master.
function _failed_result(r::_SolveOne, id, spec, extrainfo)
  ncounters = length(r.f_counters) + length(r.fnls_counters)
  problem_info = Any[id, _spec_name(spec), 0, 0, 0]
  return _result(id, :exception, _failed_row(r.solver_name, problem_info, ncounters, extrainfo))
end

# Build the statistics from the results of all problems, in any order, exactly as the original
# sequential loop does.
function _assemble(results, names, types; prune::Bool = true)
  ncounters = length(names) - 13 # 12 leading columns, the counters, then :extrainfo
  sorted = sort!(collect(Any, results), by = r -> r.id)
  stats = DataFrame(names .=> [T[] for T in types])

  # solver_specific columns: the scalar entries of the first solved problem
  specific = Symbol[]
  first_solved = findfirst(r -> r.kind === :solved, sorted)
  if first_solved !== nothing
    for (k, v) in sorted[first_solved].specific
      if !(typeof(v) <: AbstractVector)
        insertcols!(stats, ncol(stats) + 1, k => Vector{Union{typeof(v), Missing}}())
        push!(specific, k)
      end
    end
  end

  for r in sorted
    prune && r.kind === :skipped && continue
    if r.kind === :solved
      try
        d = Dict(r.specific)
        push!(stats, [r.row; [d[k] for k in specific]])
      catch e
        # a missing key or a vector value: the problem becomes an exception, as before
        @error "caught exception" e
        row = _failed_row(r.row[1], r.row[2:6], ncounters, string(e))
        push!(stats, [row; fill(missing, length(specific))])
      end
    else
      push!(stats, [r.row; fill(missing, length(specific))])
    end
  end
  return stats
end

# Runners. `run_one(id, problem)` returns a result, `logr(result)` logs its row.

function _run(::SerialRunner, run_one, logr, problems)
  results = Any[]
  for (id, problem) in enumerate(problems)
    r = run_one(id, problem)
    push!(results, r)
    logr(r)
  end
  return results
end

function _run(runner::ThreadedRunner, run_one, logr, problems)
  results = Any[]
  lk = ReentrantLock()
  # Unbuffered: the producer builds the next problem only when a task takes the previous one,
  # so problems are built one at a time and at most `ntasks + 1` are open at once.
  ch = Channel{Any}(0)
  tasks = Task[Threads.@spawn _produce!(ch, problems)]
  for _ = 1:(runner.ntasks)
    push!(tasks, Threads.@spawn _consume!(ch, run_one, logr, results, lk))
  end
  _wait_all(tasks)
  return results
end

function _produce!(ch, problems)
  try
    for (id, problem) in enumerate(problems)
      _guard_threads(problem)
      try
        put!(ch, (id, problem))
      catch e
        # a task failed and closed the channel: this problem will not be solved
        finalize(problem)
        e isa InvalidStateException && return nothing
        rethrow()
      end
    end
  finally
    close(ch)
  end
  return nothing
end

function _consume!(ch, run_one, logr, results, lk)
  try
    for (id, problem) in ch
      r = run_one(id, problem)
      lock(lk) do
        push!(results, r)
        logr(r)
      end
    end
  catch
    close(ch) # stops the producer and the other tasks
    rethrow()
  end
  return nothing
end

# Wait for every task, then throw the first failure as its original exception.
function _wait_all(tasks)
  failure = nothing
  for t in tasks
    try
      wait(t)
    catch e
      failure === nothing && (failure = e)
    end
  end
  failure === nothing && return nothing
  if failure isa TaskFailedException
    # The exception thrown below loses the task's backtrace, so show the task's whole exception
    # stack first. `showerror` of a `TaskFailedException` prints `current_exceptions(task)`.
    println(stderr, "A task of the threaded run failed:")
    showerror(stderr, failure)
    println(stderr)
  end
  throw(_task_exception(failure))
end

_task_exception(e::TaskFailedException) = _task_exception(e.task.exception)
_task_exception(e) = e

# Native-model guard. Models that wrap a native library through a pointer field (CUTEst's
# `CUTEstModel`, AmplNLReader's `AmplModel`) cannot be used from several threads, and cannot be
# sent to another process: Julia serializes every `Ptr` as `C_NULL`, without any error.
_wraps_native(problem) = any(T -> T <: Ptr, fieldtypes(typeof(problem)))

function _guard_threads(problem)
  _wraps_native(problem) || return nothing
  T = typeof(problem)
  finalize(problem)
  throw(
    ArgumentError(
      "$T wraps a native library through a pointer field and cannot be solved from " *
      "several threads. Use `distributed` instead (after `using Distributed`) with the " *
      "problems given as a generator, e.g. `distributed = 4` and " *
      "`(CUTEstModel(p; decode = false) for p in problem_names)`.",
    ),
  )
end

function _guard_send(problem)
  _wraps_native(problem) || return nothing
  T = typeof(problem)
  finalize(problem)
  throw(
    ArgumentError(
      "$T wraps a native library through a pointer field and cannot be sent to another " *
      "process (pointers arrive as null). Pass the problems as a generator so that each " *
      "worker builds its own models, e.g. " *
      "`(CUTEstModel(p; decode = false) for p in problem_names)`.",
    ),
  )
end

# Called on each new worker by the Distributed extension.
function _worker_init()
  LinearAlgebra.BLAS.set_num_threads(1)
  return nothing
end

# Hook for the Distributed extension, called as `_with_distributed(f, distributed, worker_setup)`
# where `f` receives a runner. The extension adds methods for `Integer` and `AbstractWorkerPool`.
function _with_distributed(f, distributed, worker_setup)
  if Base.get_extension(SolverBenchmark, :SolverBenchmarkDistributedExt) === nothing
    throw(
      ArgumentError(
        "`distributed` requires `using Distributed`, which loads the parallel extension " *
        "of SolverBenchmark",
      ),
    )
  end
  throw(
    ArgumentError(
      "invalid value for `distributed`: $(repr(distributed)); expected a nonnegative " *
      "integer or an `AbstractWorkerPool`",
    ),
  )
end

# Keywords `threads` and `distributed` of `solve_problems` and `bmark_solvers`.

_is_distributed(distributed) = !(distributed isa Integer && distributed == 0)

function _check_parallel_keywords(threads, distributed)
  threads >= 0 || throw(ArgumentError("`threads` must be nonnegative, got $threads"))
  if distributed isa Integer && distributed < 0
    throw(ArgumentError("`distributed` must be nonnegative, got $distributed"))
  end
  if threads > 1 && _is_distributed(distributed)
    throw(ArgumentError("`threads` and `distributed` cannot be combined: choose one of them"))
  end
  return nothing
end

function _local_runner(threads::Integer)
  threads <= 1 && return SerialRunner()
  nthreads = Threads.nthreads()
  if threads > nthreads
    @warn "`threads = $threads` but Julia was started with $nthreads thread(s), and threads " *
          "cannot be added at runtime: using $nthreads task(s). Start Julia with " *
          "`julia -t $threads` to use $threads threads."
  end
  return ThreadedRunner(min(threads, nthreads))
end
