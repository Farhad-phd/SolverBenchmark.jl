module SolverBenchmarkDistributedExt

# Implements the `distributed` keyword of `solve_problems` and `bmark_solvers`. Everything sent
# to the workers (`_RemoteJob`, `_SolveOne`, ...) is defined in SolverBenchmark itself, not here.

using Distributed
using SolverBenchmark
import SolverBenchmark:
  AbstractRunner, _RemoteJob, _failed_result, _for_workers, _guard_send, _run, _with_distributed

const UNDEFVAR_HINT =
  "Functions defined by name in your script are not sent to the workers automatically " *
  "(anonymous functions are). Names from packages or submodules loaded in `Main` are defined " *
  "on the workers only if the package or submodule is bound in `Main` under its own name, as " *
  "after `using CUTEst` or `using OptimizationProblems.ADNLPProblems`; names brought in " *
  "another way (`import`, deeper submodules, ...) are not. Define or load the missing names " *
  "on the workers with `worker_setup`, e.g. `worker_setup = \"my_definitions.jl\"`, " *
  "`worker_setup = quote ... end` or " *
  "`worker_setup = :(using OptimizationProblems.ADNLPProblems)`."

# Solves the problems on the workers of `pool`, one problem per remote call.
struct DistributedRunner{P <: AbstractWorkerPool} <: AbstractRunner
  pool::P
end

function _run(runner::DistributedRunner, run_one, logr, problems)
  if problems isa Base.Generator
    # Each worker builds its model from the generator's function and argument, so models that
    # wrap a native library work. Ids match the ids of a sequential run.
    make = problems.f
    items = collect(enumerate(problems.iter))
  else
    make = identity
    items = Any[]
    for (id, problem) in enumerate(problems)
      _guard_send(problem)
      push!(items, (id, problem))
    end
  end
  isempty(items) && return Any[]
  job = _RemoteJob(make, _for_workers(run_one))
  pool = runner.pool
  results = try
    asyncmap(items; ntasks = max(1, length(pool))) do item
      _solve_remote(job, pool, item, run_one, logr)
    end
  catch e
    _undefvar_hint(e)
    rethrow()
  end
  if any(r -> r.kind === :exception && startswith(r.row[end], "UndefVarError"), results)
    @warn "some problems failed with an `UndefVarError` on the workers. " * UNDEFVAR_HINT
  end
  return results
end

# Solve one item on a worker of `pool`. The remote call is wrapped here, on the master, rather
# than through `pmap(...; on_error)`: with `batch_size = 1`, `pmap` wraps `on_error` around the
# function before sending it, so `on_error` runs on the worker and never sees the worker die.
function _solve_remote(job, pool, item, run_one, logr)
  r = try
    remotecall_fetch(job, pool, item)
  catch e
    _worker_died(e) || rethrow()
    id, spec = item
    msg = "a worker died while solving problem $id; it is recorded as an exception"
    @error msg exception = e
    _failed_result(run_one, id, spec, "worker died: $(_root_exception(e))")
  end
  logr(r)
  return r
end

_root_exception(e::CapturedException) = _root_exception(e.ex)
_root_exception(e::RemoteException) = _root_exception(e.captured)
_root_exception(e::TaskFailedException) = _root_exception(e.task.exception)
_root_exception(e::CompositeException) =
  isempty(e.exceptions) ? e : _root_exception(first(e.exceptions))
_root_exception(e) = e

_worker_died(e) = _root_exception(e) isa ProcessExitedException

function _undefvar_hint(e)
  root = _root_exception(e)
  root isa UndefVarError || return nothing
  throw(ErrorException("a worker raised $(sprint(showerror, root)). " * UNDEFVAR_HINT))
end

"""
    with_workers(f, n, worker_setup = nothing)

Start `n` worker processes, prepare them for `solve_problems`, and return `f(pool)` where `pool`
is a `WorkerPool` of these workers. The workers are removed when `f` returns or throws; workers
that existed before are not touched.

Each worker runs with one thread and one BLAS thread, uses the active project and the
`LOAD_PATH` of this process, and loads `SolverBenchmark`. It then loads with `using` every
package bound in `Main` under its own name (as after `using CUTEst`), then every direct
submodule of a package bound in `Main` under its own name (as after
`using OptimizationProblems.ADNLPProblems`). Finally it runs `worker_setup`: an `Expr`
evaluated in `Main`, or the path of a file included in `Main`.
"""
function with_workers(f, n::Integer, worker_setup = nothing)
  n >= 1 || throw(ArgumentError("the number of workers must be positive, got $n"))
  setup = _setup_expr(worker_setup)
  flags = `--threads=1`
  project = Base.active_project()
  if project !== nothing
    flags = `$flags --project=$project`
  end
  ids = addprocs(n; exeflags = flags)
  try
    _init_workers(ids, setup)
    return f(WorkerPool(ids))
  finally
    _remove_workers(ids)
  end
end

function _setup_expr(worker_setup)
  worker_setup === nothing && return nothing
  worker_setup isa Expr && return worker_setup
  if worker_setup isa AbstractString
    path = abspath(worker_setup)
    isfile(path) || throw(ArgumentError("`worker_setup` file not found: $path"))
    return :(include($path))
  end
  throw(
    ArgumentError(
      "`worker_setup` must be an `Expr` or the path of a file, got $(typeof(worker_setup))",
    ),
  )
end

# Whether module `m` is bound in module `main` under its own name.
_bound_in(main::Module, m::Module) =
  isdefined(main, nameof(m)) && getglobal(main, nameof(m)) === m

# Modules to load with `using` on the workers so that their `Main` has the names of `main`:
# 1. the modules of `loaded` bound in `main` under their own name, e.g. after `using CUTEst`;
# 2. then the direct submodules of the modules of `loaded` bound in `main` under their own name,
#    e.g. after `using OptimizationProblems.ADNLPProblems`.
# `loaded` is `values(Base.loaded_modules)` in practice; a test passes its own modules.
function _modules_to_load(main::Module, loaded)
  candidates = Module[m for m in loaded if !(nameof(m) in (:Core, :Base, :Main))]
  packages = Module[m for m in candidates if _bound_in(main, m)]
  submodules = Module[]
  for p in candidates, name in names(p; all = true)
    # skip deprecated bindings: reading them prints a deprecation warning
    (isdefined(p, name) && !Base.isdeprecated(p, name)) || continue
    s = getglobal(p, name)
    if s isa Module && s !== p && parentmodule(s) === p && _bound_in(main, s)
      push!(submodules, s)
    end
  end
  sort!(unique!(packages), by = nameof)
  sort!(unique!(submodules), by = fullname)
  return [packages; submodules]
end

function _init_workers(ids, setup)
  load_path = copy(LOAD_PATH)
  Distributed.remotecall_eval(Main, ids, :(append!(empty!(LOAD_PATH), $load_path)))
  Distributed.remotecall_eval(Main, ids, :(using SolverBenchmark))
  for m in _modules_to_load(Main, collect(values(Base.loaded_modules)))
    path = fullname(m) # (:CUTEst,) or (:OptimizationProblems, :ADNLPProblems)
    try
      Distributed.remotecall_eval(Main, ids, Expr(:using, Expr(:., path...)))
    catch e
      @warn "could not load $(join(path, '.')) on the workers" exception = e
    end
  end
  Distributed.remotecall_eval(Main, ids, :(SolverBenchmark._worker_init()))
  setup === nothing || Distributed.remotecall_eval(Main, ids, setup)
  return nothing
end

# Remove the workers in `ids` that are still alive. A worker that died is already gone.
function _remove_workers(ids)
  alive = filter(in(procs()), ids)
  isempty(alive) && return nothing
  try
    rmprocs(alive)
  catch e
    @warn "could not remove workers $alive" exception = e
  end
  return nothing
end

function _with_distributed(f, n::Integer, worker_setup)
  return with_workers(pool -> f(DistributedRunner(pool)), n, worker_setup)
end

function _with_distributed(f, pool::AbstractWorkerPool, worker_setup)
  if worker_setup !== nothing
    throw(
      ArgumentError(
        "`worker_setup` cannot be used with an existing worker pool: prepare its workers " *
        "yourself, e.g. with `@everywhere`",
      ),
    )
  end
  return f(DistributedRunner(pool))
end

end
