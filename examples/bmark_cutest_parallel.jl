# Benchmark CUTEst problems on 4 worker processes.
#
# `CUTEstModel(p)` decodes and compiles problem `p` at each construction, in a folder shared by
# all processes and with file names that depend only on the problem name and precision. So the
# problems are decoded and compiled once, sequentially, and the workers build the models with
# `decode = false`, which only reads the compiled files. On Windows, this is also required
# because a library loaded by one process cannot be overwritten by another.
using Distributed, CUTEst, JSOSolvers, SolverBenchmark

problem_names = CUTEst.select_sif_problems(max_con = 0, min_var = 50, max_var = 80)
# one size per SIF name: the sizes of a problem share one compiled library
@assert allunique(problem_names)
for p in problem_names  # decode and compile once, sequentially, before the benchmark
  sifdecoder(p)
  build_libsif(p)
end

# A generator, so that each worker builds its own models: CUTEst models cannot be sent to
# another process, and cannot be used with `threads`.
problems = (CUTEstModel(p; decode = false) for p in problem_names)  # only reads files

solvers = Dict(
  :trunk => nlp -> trunk(nlp, max_time = 60.0),
  :lbfgs => nlp -> lbfgs(nlp, max_time = 60.0),
)

# the 4 workers are started once for both solvers, and stopped at the end
stats = bmark_solvers(solvers, problems; distributed = 4)

# timings of a parallel run are distorted by contention: compare costs based on counters
performance_profile(stats, df -> df.neval_obj)
