# Tests of the `threads` and `distributed` keywords of `solve_problems` and `bmark_solvers`.
# Runs from runtests.jl or on its own (after `TestEnv.activate()`), and can be included several
# times in one session.
using Test
using Distributed
using Random
using Logging
using LinearAlgebra
using DataFrames
using NLPModels, ADNLPModels
using SolverCore
using SolverBenchmark

include(joinpath(@__DIR__, "reference_solve_problems.jl"))
include(joinpath(@__DIR__, "parallel_helpers.jl"))

helpers_path = joinpath(@__DIR__, "parallel_helpers.jl")
ext = Base.get_extension(SolverBenchmark, :SolverBenchmarkDistributedExt)
procs_before = procs()

quietly(f) = with_logger(f, NullLogger())

# Same statistics, except for timings and for the thread and worker that solved each problem.
function check_same(a, b)
  @test names(a) == names(b)
  names(a) == names(b) || return nothing
  @test [eltype(c) for c in eachcol(a)] == [eltype(c) for c in eachcol(b)]
  ignored = ("elapsed_time", "thread", "worker")
  differing = [c for c in names(a) if !(c in ignored) && !isequal(a[!, c], b[!, c])]
  @test differing == String[]
  @test isfinite.(a.elapsed_time) == isfinite.(b.elapsed_time)
  return nothing
end

# The `MethodError` raised for a vector value shows the world age, which grows whenever a method
# is defined. Strip it so that the comparison does not depend on when each run happened.
function strip_world_age(df)
  df = copy(df)
  df.extrainfo = [replace(s, r"0x[0-9a-f]{16}" => "0x…") for s in df.extrainfo]
  return df
end

# `check_same` for the statistics of a scenario.
function check_scenario(sc, a, b)
  if isempty(sc.vector_key)
    check_same(a, b)
  else
    check_same(strip_world_age(a), strip_world_age(b))
  end
  return nothing
end

function scenario(
  name,
  N;
  fail = Int[],
  skip = Int[],
  prune = true,
  missing_key = Int[],
  vector_key = Int[],
)
  return (
    name = name,
    N = N,
    fail = Set{Int}(fail),
    skip = Set{Int}(skip),
    prune = prune,
    missing_key = Set{Int}(missing_key),
    vector_key = Set{Int}(vector_key),
  )
end

function random_scenario(seed)
  rng = MersenneTwister(seed)
  N = rand(rng, 1:40)
  fail = [k for k = 1:N if rand(rng) < 0.2]
  skip = [k for k = 1:N if rand(rng) < 0.2]
  prune = rand(rng, Bool)
  return scenario("random, seed = $seed", N; fail = fail, skip = skip, prune = prune)
end

scenario_solver(sc) =
  make_solver(fail = sc.fail, missing_key = sc.missing_key, vector_key = sc.vector_key)

# The problems are passed as a generator.
function run_scenario(sc; kwargs...)
  solver = scenario_solver(sc)
  skip = sc.skip
  skipif = nlp -> problem_index(nlp) in skip
  problems = (make_problem(k) for k = 1:(sc.N))
  return quietly() do
    solve_problems(solver, "solver", problems; skipif = skipif, prune = sc.prune, kwargs...)
  end
end

function run_scenario_reference(sc)
  solver = scenario_solver(sc)
  skip = sc.skip
  skipif = nlp -> problem_index(nlp) in skip
  problems = (make_problem(k) for k = 1:(sc.N))
  return quietly() do
    reference_solve_problems(solver, "solver", problems; skipif = skipif, prune = sc.prune)
  end
end

base_scenarios = [
  scenario("empty list", 0),
  scenario("first problem fails", 6; fail = [1]),
  scenario("all problems fail", 5; fail = 1:5),
  scenario("first problem skipped, prune = false", 6; skip = [1], prune = false),
  scenario("first problem skipped, prune = true", 6; skip = [1], prune = true),
  [random_scenario(seed) for seed = 1:25]...,
]
specific_scenarios = [
  scenario("a later problem misses a solver_specific key", 6; missing_key = [4]),
  scenario("a later problem returns a vector for a solver_specific key", 6; vector_key = [4]),
]
all_scenarios = [base_scenarios; specific_scenarios]

vector_problems = [make_problem(k) for k = 1:12]
vector_solver = make_solver(fail = Set([3]))
run_vector(; kwargs...) =
  quietly(() -> solve_problems(vector_solver, "solver", vector_problems; kwargs...))

# A fake package with two submodules, and fake `Main` modules that load them in different ways,
# all inside an anonymous host module, so that including this file again replaces nothing.
# Like `LinearAlgebra` with `BLAS`, the package exports the name of its submodule.
function fake_modules()
  host = Module(:FakeHost)
  Core.eval(host, :(module FakePackage
  export FakeSubmodule
  module FakeSubmodule
  export fake_constructor
  fake_constructor() = 1
  end
  module OtherSubmodule end
  end))
  Core.eval(host, :(module MainWithBoth
  using ..FakePackage
  using ..FakePackage.FakeSubmodule
  end))
  Core.eval(host, :(module MainWithPackage
  using ..FakePackage
  end))
  Core.eval(host, :(module MainWithSubmodule
  using ..FakePackage.FakeSubmodule
  end))
  Core.eval(host, :(module MainWithRenamedSubmodule
  import ..FakePackage.FakeSubmodule as Renamed
  end))
  Core.eval(host, :(module MainWithLocalModule
  using ..FakePackage
  module LocalModule
  export local_value
  local_value = 1
  end
  using .LocalModule
  end))
  return host
end

bmark_dict() = Dict(:a => make_solver(fail = Set([2])), :b => make_solver(fail = Set([1, 5])))
bmark_problems() = (make_problem(k) for k = 1:10)
run_bmark(; kwargs...) = quietly(() -> bmark_solvers(bmark_dict(), bmark_problems(); kwargs...))

@testset "parallel runs" begin
  @test ext !== nothing

  serial = Dict(sc.name => run_scenario(sc) for sc in all_scenarios)
  serial_vector = run_vector()
  serial_bmark = run_bmark()

  @testset "sequential run unchanged: $(sc.name)" for sc in all_scenarios
    check_scenario(sc, serial[sc.name], run_scenario_reference(sc))
  end

  @testset "sequential run unchanged: Vector of models" begin
    reference = quietly(() -> reference_solve_problems(vector_solver, "solver", vector_problems))
    check_same(serial_vector, reference)
  end

  @testset "sequential run unchanged: bmark_solvers" begin
    for (name, solver) in bmark_dict()
      check_same(
        serial_bmark[name],
        quietly(() -> reference_solve_problems(solver, name, bmark_problems())),
      )
    end
  end

  @testset "threads = 4: $(sc.name)" for sc in all_scenarios
    check_scenario(sc, run_scenario(sc; threads = 4), serial[sc.name])
  end

  @testset "threads = 4: Vector of models" begin
    check_same(run_vector(threads = 4), serial_vector)
  end

  @testset "threads = 4: bmark_solvers" begin
    stats = run_bmark(threads = 4)
    @test sort(collect(keys(stats))) == sort(collect(keys(serial_bmark)))
    for name in keys(serial_bmark)
      check_same(stats[name], serial_bmark[name])
    end
  end

  # One pool of workers for all the comparisons with sequential runs.
  ext.with_workers(2, helpers_path) do pool
    @testset "workers run with one thread and one BLAS thread" begin
      for w in collect(pool.workers)
        @test remotecall_fetch(Threads.nthreads, w) == 1
        @test remotecall_fetch(LinearAlgebra.BLAS.get_num_threads, w) == 1
      end
    end

    @testset "distributed: $(sc.name)" for sc in all_scenarios
      check_scenario(sc, run_scenario(sc; distributed = pool), serial[sc.name])
    end

    @testset "distributed: Vector of models" begin
      check_same(run_vector(distributed = pool), serial_vector)
    end

    @testset "distributed: bmark_solvers" begin
      stats = run_bmark(distributed = pool)
      @test sort(collect(keys(stats))) == sort(collect(keys(serial_bmark)))
      for name in keys(serial_bmark)
        check_same(stats[name], serial_bmark[name])
      end
    end

    @testset "distributed: solver_logger other than NullLogger" begin
      sc = scenario("solver_logger", 5; fail = [2])
      logger = ConsoleLogger(stderr, Logging.Warn)
      check_same(run_scenario(sc; distributed = pool, solver_logger = logger), run_scenario(sc))
    end

    @testset "a pointer arrives as null on a worker" begin
      p = PtrModel(1)
      p.handle = Ptr{Cvoid}(UInt(0x1234))
      @test p.handle != C_NULL
      @test remotecall_fetch(m -> m.handle == C_NULL, pool, p)
    end

    @testset "native-model guard with distributed" begin
      solver = make_solver()
      @test_throws ArgumentError quietly() do
        solve_problems(solver, "solver", [PtrModel(k) for k = 1:3]; distributed = pool)
      end
      ptr_serial = quietly(() -> solve_problems(solver, "solver", (PtrModel(k) for k = 1:3)))
      ptr_distributed = quietly() do
        solve_problems(solver, "solver", (PtrModel(k) for k = 1:3); distributed = pool)
      end
      @test all(ptr_distributed.status .== :first_order)
      check_same(ptr_distributed, ptr_serial)
    end

    @testset "worker_setup with a pool" begin
      @test_throws ArgumentError solve_problems(
        make_solver(),
        "solver",
        (make_problem(k) for k = 1:2);
        distributed = pool,
        worker_setup = helpers_path,
      )
    end

    @testset "witness: several workers used" begin
      stats = quietly() do
        problems = (make_problem(k) for k = 1:12)
        solve_problems(make_solver(delay = 0.05), "solver", problems; distributed = pool)
      end
      @test length(unique(skipmissing(stats.worker))) > 1
      @test all(!=(1), skipmissing(stats.worker))
    end
  end

  @testset "worker lifecycle" begin
    n0 = nprocs()
    solver = make_solver()

    # a run leaves the number of processes unchanged and does not touch existing workers
    pre = addprocs(1)
    stats = quietly() do
      problems = (make_problem(k) for k = 1:4)
      solve_problems(solver, "solver", problems; distributed = 2, worker_setup = helpers_path)
    end
    @test nrow(stats) == 4
    @test issubset(pre, workers())
    @test nprocs() == n0 + 1
    rmprocs(pre)
    @test nprocs() == n0

    # a run that throws also removes its workers
    @test_throws Exception quietly() do
      problems = (k == 2 ? error("planned failure") : make_problem(k) for k = 1:4)
      solve_problems(solver, "solver", problems; distributed = 2, worker_setup = helpers_path)
    end
    @test nprocs() == n0

    # bmark_solvers starts the workers once for all solvers
    stats = run_bmark(distributed = 2, worker_setup = helpers_path)
    used = unique(collect(skipmissing(vcat(stats[:a].worker, stats[:b].worker))))
    @test 1 <= length(used) <= 2
    @test all(!=(1), used)
    for name in keys(serial_bmark)
      check_same(stats[name], serial_bmark[name])
    end
    @test nprocs() == n0

    # a worker that dies is recorded as an exception and the run continues
    died = quietly() do
      problems = (make_problem(k) for k = 1:8)
      solve_problems(
        make_solver(die = Set([3])),
        "solver",
        problems;
        distributed = 2,
        worker_setup = helpers_path,
      )
    end
    reference = quietly(() -> solve_problems(solver, "solver", (make_problem(k) for k = 1:8)))
    @test nrow(died) == 8
    row3 = died[died.id .== 3, :]
    @test nrow(row3) == 1
    @test row3.status[1] === :exception
    @test occursin("worker died", row3.extrainfo[1])
    check_same(died[died.id .!= 3, :], reference[reference.id .!= 3, :])
    @test nprocs() == n0
  end

  @testset "guards and arguments" begin
    solver = make_solver()
    @test_throws ArgumentError quietly() do
      solve_problems(solver, "solver", [PtrModel(1), PtrModel(2)]; threads = 2)
    end
    @test_logs (:warn, r"julia -t") match_mode = :any solve_problems(
      solver,
      "solver",
      (make_problem(k) for k = 1:2);
      threads = Threads.nthreads() + 1,
    )
    @test_throws ArgumentError solve_problems(solver, "solver", Any[]; threads = 2, distributed = 2)
    @test_throws ArgumentError solve_problems(solver, "solver", Any[]; threads = -1)
    @test_throws ArgumentError solve_problems(solver, "solver", Any[]; distributed = -1)
    @test_throws ArgumentError solve_problems(solver, "solver", Any[]; distributed = "two")
    # `Bool <: Integer`, but `distributed = true` must not start one worker
    for value in (true, false)
      @test_throws ArgumentError solve_problems(solver, "solver", Any[]; threads = value)
      @test_throws ArgumentError solve_problems(solver, "solver", Any[]; distributed = value)
      @test_throws ArgumentError bmark_solvers(Dict(:s => solver), Any[]; threads = value)
      @test_throws ArgumentError bmark_solvers(Dict(:s => solver), Any[]; distributed = value)
    end
  end

  @testset "modules loaded on the workers" begin
    host = fake_modules()
    pkg = host.FakePackage
    sub = pkg.FakeSubmodule
    modules_to_load = ext._modules_to_load
    # the fake modules behave as intended
    @test isdefined(host.MainWithSubmodule, :fake_constructor)
    @test !isdefined(host.MainWithPackage, :fake_constructor)
    @test getglobal(host.MainWithPackage, :FakeSubmodule) === sub
    @test isdefined(host.MainWithLocalModule, :local_value)

    @test modules_to_load(host.MainWithBoth, [pkg]) == [pkg, sub]
    # `using FakePackage` binds the exported name `FakeSubmodule`, but `Main` is not `using`
    # the submodule: it is not loaded
    @test modules_to_load(host.MainWithPackage, [pkg]) == [pkg]
    # the submodule is loaded even when its package is not bound in `Main`
    @test modules_to_load(host.MainWithSubmodule, [pkg]) == [sub]
    # `import ... as Renamed`: neither bound under its own name nor `using`'d
    @test modules_to_load(host.MainWithRenamedSubmodule, [pkg]) == Module[]
    # a module defined in `Main` itself is not loaded, even if `Main` is `using` it
    @test modules_to_load(host.MainWithLocalModule, [pkg]) == [pkg]
    # modules outside the loaded packages, and Core and Base, are ignored
    @test modules_to_load(host.MainWithSubmodule, Module[]) == Module[]
    @test modules_to_load(host.MainWithBoth, [Core, Base]) == Module[]
  end

  @testset "witness: several threads used" begin
    if Threads.nthreads() > 1
      stats = quietly() do
        problems = (make_problem(k) for k = 1:16)
        solve_problems(
          make_solver(delay = 0.05),
          "solver",
          problems;
          threads = Threads.nthreads(),
        )
      end
      @test length(unique(skipmissing(stats.thread))) > 1
    else
      @warn "Julia runs with one thread: the threaded witness test is skipped. " *
            "Start Julia with `-t 2` or more to run it."
    end
  end

  @testset "without the extension" begin
    # A fresh process that does not load Distributed. The code is written to a file, since code
    # passed on the command line does not survive the quoting rules of every platform.
    script = tempname() * ".jl"
    write(
      script,
      """
      using SolverBenchmark, ADNLPModels
      loaded = any(m -> nameof(m) === :Distributed, values(Base.loaded_modules))
      println("DISTRIBUTED_LOADED=", loaded)
      nlp = ADNLPModel(x -> sum(x .^ 2), ones(2))
      message = try
        solve_problems(p -> nothing, "solver", [nlp]; distributed = 2)
        "no error"
      catch e
        e isa ArgumentError ? "ArgumentError: " * e.msg : "unexpected: " * sprint(showerror, e)
      end
      println("DISTRIBUTED_ERROR=", message)
      """,
    )
    julia = `$(Base.julia_cmd()) --startup-file=no`
    project = Base.active_project()
    cmd = project === nothing ? `$julia $script` : `$julia --project=$project $script`
    output = try
      read(cmd, String)
    finally
      rm(script; force = true)
    end
    @test occursin("DISTRIBUTED_LOADED=false", output)
    @test occursin(r"DISTRIBUTED_ERROR=ArgumentError: .*using Distributed", output)
  end

  @testset "all workers removed" begin
    leftover = setdiff(procs(), procs_before)
    @test isempty(leftover)
    isempty(leftover) || rmprocs(leftover)
  end
end
