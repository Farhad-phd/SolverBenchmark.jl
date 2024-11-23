using Base.Threads
using DataFrames

export bmark_solvers
"""
    bmark_solvers(solvers :: Dict{Symbol, Any}, args...; kwargs...)

Run a set of solvers on a set of problems. If `JULIA_NUM_THREADS` is set to a number greater than 1, 
the solvers will be run in parallel using `Threads.@threads`.

Note: The `@threads` macro requires that the number of threads be set before the Julia process starts, using the `-t` or `--threads`
command-line argument, or the `JULIA_NUM_THREADS` environment variable.

#### Arguments
* `solvers`: a dictionary of solvers to which each problem should be passed
* other positional arguments accepted by `solve_problems`, except for a solver name

#### Keyword arguments
Any keyword argument accepted by `solve_problems`

#### Return value
A Dict{Symbol, AbstractExecutionStats} of statistics.
"""

function bmark_solvers(solvers::Dict{Symbol, <:Any}, args...; kwargs...)
  stats = Dict{Symbol, DataFrame}()
  # Initialize the lock for thread-safe updates
  lock = ReentrantLock()
  key_array = collect(keys(solvers))  # Convert keys to an array for indexing

  Threads.@threads for i in eachindex(key_array) 
    name = key_array[i]
    solver = solvers[name]
    @info "Running solver $name on thread $(Threads.threadid())"
    result = solve_problems(solver, name, args...; kwargs...)
    # Ensure thread-safe access to `stats`
    lock(lock) do
      stats[name] = result
    end
  end
  return stats
end


"""
    bmark_solvers_parallel(solvers :: Dict{Symbol, Any}, args...; kwargs...)

Run a set of solvers on a set of problems in parallel.

#### Arguments
* `solvers`: a dictionary of solvers to which each problem should be passed.
* `args...`: other positional arguments accepted by `solve_problems`, except for a solver name.

#### Keyword arguments
Any keyword argument accepted by `solve_problems`.

#### Return value
A `Dict{Symbol, DataFrame}` of execution statistics, with each solver's results stored under its corresponding key.
"""


function bmark_solvers_parallel(solvers::Dict{Symbol, <:Any}, args...; kwargs...)
  stats = Dict{Symbol, DataFrame}()

  # Collect solvers in a vector so we can iterate them in parallel
  solver_keys = collect(keys(solvers))
  
  @threads for i in 1:length(solver_keys)
      name = solver_keys[i]
      solver = solvers[name]
      @info "running solver $name on thread $(threadid())"
      stats[name] = solve_problems(solver, name, args...; kwargs...)
  end
  return stats
end
