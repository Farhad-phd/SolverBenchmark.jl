# Test problems and solvers for test_parallel.jl. This file is also included in `Main` on the
# worker processes through `worker_setup`, so it loads everything it needs. It is safe to
# include several times in one session.
using Distributed
using LinearAlgebra
using NLPModels, ADNLPModels
using SolverCore

# index `k` of a problem named "p_k" or "ptr_k"
problem_index(nlp) = parse(Int, last(split(get_name(nlp), '_')))

# Deterministic in `k`, named "p_k"; every 5th problem is an `ADNLSModel`.
function make_problem(k::Integer)
  n = 2 + k % 3
  x0 = [k / 10 + i for i = 1:n]
  if k % 5 == 0
    return ADNLSModel(x -> x .- k, x0, n, name = "p_$k")
  else
    return ADNLPModel(x -> sum((x .- k) .^ 2) + k, x0, name = "p_$k")
  end
end

# Deterministic solver. For problem `k`, it sleeps `delay * rand()` seconds, kills its process
# if `k ∈ die`, throws if `k ∈ fail`, evaluates the objective and the gradient at `x0`, and sets
# solver_specific `:k2 => k^2` (omitted if `k ∈ missing_key`, a vector if `k ∈ vector_key`),
# `:thread => Threads.threadid()` and `:worker => myid()`.
function make_solver(;
  fail = Set{Int}(),
  delay = 0.0,
  die = Set{Int}(),
  missing_key = Set{Int}(),
  vector_key = Set{Int}(),
)
  return function (nlp; kwargs...)
    t0 = time()
    k = problem_index(nlp)
    delay > 0 && sleep(delay * rand())
    k in die && exit()
    k in fail && error("planned failure on problem $k")
    x = copy(nlp.meta.x0)
    fx = obj(nlp, x)
    gx = grad(nlp, x)
    stats = GenericExecutionStats(nlp)
    set_status!(stats, :first_order)
    set_objective!(stats, fx)
    set_residuals!(stats, 0.0, norm(gx))
    set_iter!(stats, k)
    set_time!(stats, time() - t0)
    if !(k in missing_key)
      set_solver_specific!(stats, :k2, k in vector_key ? [k, k] : k^2)
    end
    set_solver_specific!(stats, :thread, Threads.threadid())
    set_solver_specific!(stats, :worker, myid())
    return stats
  end
end

# Model holding a pointer, like the models that wrap a native library (CUTEst, AMPL).
mutable struct PtrModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  counters::Counters
  handle::Ptr{Cvoid}
end

PtrModel(k::Integer) =
  PtrModel(NLPModelMeta(2, x0 = fill(Float64(k), 2), name = "ptr_$k"), Counters(), C_NULL)

function NLPModels.obj(nlp::PtrModel, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  return sum(abs2, x)
end

function NLPModels.grad!(nlp::PtrModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(nlp, :neval_grad)
  g .= 2 .* x
  return g
end
