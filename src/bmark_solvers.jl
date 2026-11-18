export bmark_solvers

"""
    bmark_solvers(solvers :: Dict{Symbol,Any}, args...; kwargs...)

Run a set of solvers on a set of problems.

#### Arguments
* `solvers`: a dictionary of solvers to which each problem should be passed
* other positional arguments accepted by `solve_problems`, except for a solver name

#### Keyword arguments
Any keyword argument accepted by `solve_problems`, including `threads`, `distributed` and
`worker_setup`. The solvers run one after the other; the problems of each solver are solved in
parallel if requested. With `distributed = n`, the `n` workers are started once and used by all
solvers. See the "Parallel runs" section of `solve_problems`.

Keywords named `threads`, `distributed` and `worker_setup` are not forwarded to the solvers; to
pass a solver option with one of these names, wrap the solver, e.g.
`:mysolver => nlp -> mysolver(nlp; threads = 4)`.

#### Return value
A Dict{Symbol, AbstractExecutionStats} of statistics.
"""
function bmark_solvers(
  solvers::Dict{Symbol, <:Any},
  args...;
  threads::Integer = 0,
  distributed = 0,
  worker_setup = nothing,
  kwargs...,
)
  _check_parallel_keywords(threads, distributed)
  if _is_distributed(distributed)
    # start the workers once and run every solver on them
    return _with_distributed(distributed, worker_setup) do runner
      _bmark_solvers(solvers, runner, args...; kwargs...)
    end
  end
  return _bmark_solvers(solvers, _local_runner(threads), args...; kwargs...)
end

function _bmark_solvers(solvers, runner, args...; kwargs...)
  stats = Dict{Symbol, DataFrame}()
  for (name, solver) in solvers
    @info "running solver $name"
    stats[name] = _solve_problems(solver, name, args..., runner; kwargs...)
  end
  return stats
end
