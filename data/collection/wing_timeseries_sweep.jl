# Parametric sweep over wing designs that logs the full per-timestep transient
# of each unsteady FLOWUnsteady simulation (see wing_sim.jl for the solver
# settings and reference quantities) as the wake develops from t=0 to steady
# state. Steady-state coefficients are extracted downstream from the tail of
# each transient (src/datasets/preprocess.py).
#
# Output: wing_timeseries_data_<fidelity>_n<nsteps>.csv, long format (one row
# per sample+timestep).
# CL and CD are based on the planform area S_ref, and Cm on S_ref and the mean
# aerodynamic chord `mac` about the MAC quarter chord. `converged` is 1 if the
# simulation completed (0 rows are placeholders for failed samples).
#
# The Latin hypercube rows are in random order, so any prefix of the samples
# is itself a uniform random sample of the design space. Rerunning resumes:
# samples that completed in the output file are skipped, failed ones retried.
#
# Usage:
#   julia -t 16 --project=. wing_timeseries_sweep.jl
#
# Optional sharding across processes: shard k of N runs samples k, k+N, k+2N,
# ... and writes its own CSV. Concatenate the shard files after all finish.
#   SHARD=2 NSHARDS=4 julia -t 4 --project=. wing_timeseries_sweep.jl
#
# FIDELITY=low selects the Weber-example preset (base training data); NSTEPS
# overrides the number of time steps.

using Random
using Printf
include(joinpath(@__DIR__, "wing_sim.jl"))

# Sweep configuration
num_samples     = 500

AOA_range       = (0.0, 12.0)       # (deg) angle of attack
ar_range        = (3.0, 10.0)       # Aspect ratio (b / c_tip)
tr_range        = (0.3, 1.0)        # Taper ratio (c_tip / c_root)
lambda_range    = (0.0, 50.0)       # (deg) leading-edge sweep angle
gamma_range     = (-5.0, 10.0)      # (deg) dihedral angle
twist_tip_range = (-5.0, 5.0)       # (deg) tip twist (washout if negative), straight LE/TE from the untwisted root
magVinf_range   = (20.0, 80.0)      # (m/s) freestream velocity

# Fixed parameters
b               = 2.489             # (m) span length
rho             = 1.225             # (kg/m^3) air density (sea level ISA)
mu              = 1.81e-5           # (Pa*s) dynamic viscosity of air at ~15C

fidelity        = get(ENV, "FIDELITY", "high")
nsteps          = parse(Int, get(ENV, "NSTEPS", "200"))

# Execution / output
nshards         = parse(Int, get(ENV, "NSHARDS", "1"))
shard           = parse(Int, get(ENV, "SHARD", "1"))
@assert 1 <= shard <= nshards "SHARD must be in 1:NSHARDS"

run_tag         = "$(fidelity)_n$(nsteps)"   # never resume into a file of other settings
output_file     = nshards == 1 ? "wing_timeseries_data_$(run_tag).csv" :
                  "wing_timeseries_data_$(run_tag)_shard$(shard)of$(nshards).csv"
seed            = 42


function latin_hypercube(ranges, n_samples; rng=Random.default_rng())
    n_params = length(ranges)
    samples = zeros(n_samples, n_params)
    for j in 1:n_params
        lo, hi = ranges[j]
        perm = randperm(rng, n_samples)
        for i in 1:n_samples
            lo_stratum = (perm[i] - 1) / n_samples
            hi_stratum = perm[i] / n_samples
            u = lo_stratum + rand(rng) * (hi_stratum - lo_stratum)
            samples[i, j] = lo + u * (hi - lo)
        end
    end
    return samples
end

csv_row(vals) = join([@sprintf("%.8g", v) for v in vals], ",")


# Generate sample points
Random.seed!(seed)

println("Generating $num_samples Latin Hypercube samples...")
ranges = [AOA_range, ar_range, tr_range, lambda_range,
          gamma_range, twist_tip_range, magVinf_range]
samples = latin_hypercube(ranges, num_samples)

configs = [(AOA       = samples[i, 1],
            ar        = samples[i, 2],
            tr        = samples[i, 3],
            lambda    = samples[i, 4],
            gamma     = samples[i, 5],
            twist_tip = samples[i, 6],
            magVinf   = samples[i, 7])
           for i in 1:num_samples]

const CSV_HEADER = join([
    "sample_id",
    "AOA", "ar", "tr", "lambda", "gamma", "twist_tip", "magVinf",
    "qinf", "S_ref", "mac", "c_root", "c_tip", "Re_mac",
    "step", "t", "CL", "CD", "Cm",
    "converged"
], ",")

# Resume: skip samples already written by a previous run
done_ids = Set{Int}()
if isfile(output_file)
    lines = readlines(output_file)
    @assert !isempty(lines) && lines[1] == CSV_HEADER "$output_file has a different header; move it away first"
    for line in lines[2:end]
        fields = split(line, ",")
        # failed samples (converged = 0 placeholders) are retried
        fields[end] == "1" && push!(done_ids, round(Int, parse(Float64, fields[1])))
    end
else
    open(f -> println(f, CSV_HEADER), output_file, "w")
end

my_ids = [i for i in 1:num_samples if mod1(i, nshards) == shard && !(i in done_ids)]

println("=" ^ 72)
println("STARTING TIME-SERIES SWEEP  ($(length(my_ids)) configurations to run, fidelity=$fidelity, nsteps=$nsteps)")
nshards > 1 && println("Shard $shard of $nshards  →  $output_file")
isempty(done_ids) || println("Resuming: $(length(done_ids)) samples already in $output_file")
println("=" ^ 72)

n_success = 0
n_fail    = 0

for idx in my_ids
    cfg = configs[idx]

    @printf("\n[%d/%d] AOA=%5.1f°  ar=%4.1f  tr=%.2f  Λ=%5.1f°  Γ=%5.1f°  twist=%5.1f°  V=%5.1f m/s\n",
            idx, num_samples,
            cfg.AOA, cfg.ar, cfg.tr, cfg.lambda, cfg.gamma, cfg.twist_tip, cfg.magVinf)
    flush(stdout)   # stdout redirected to a file is block-buffered

    pf   = planform(b, cfg.ar, cfg.tr, cfg.lambda, cfg.gamma)
    qinf = 0.5 * rho * cfg.magVinf^2
    base = [idx,
            cfg.AOA, cfg.ar, cfg.tr, cfg.lambda, cfg.gamma, cfg.twist_tip, cfg.magVinf,
            qinf, pf.S, pf.mac, pf.c_root, pf.c_tip, rho * cfg.magVinf * pf.mac / mu]

    rows = String[]
    try
        wall = @elapsed res = run_wing(; cfg.AOA, cfg.ar, cfg.tr, cfg.lambda, cfg.gamma,
                                         cfg.twist_tip, cfg.magVinf, b, rho, fidelity, nsteps)
        nrec = min(length(res.t), length(res.CL), length(res.CD), length(res.Cm))
        nrec > 0 || error("no steps logged")
        all(isfinite, vcat(res.CL, res.CD, res.Cm)) || error("non-finite coefficients")
        for k in 1:nrec
            push!(rows, csv_row(vcat(base, [k, res.t[k], res.CL[k], res.CD[k], res.Cm[k], 1])))
        end
        global n_success += 1
        @printf("       logged %d steps in %.0f s  |  final CL=%.4f  CD=%.5f  Cm=%.4f  |  removed %d blown-up, %d by size; peak strength %.2f of bound\n",
                nrec, wall, res.CL[end], res.CD[end], res.Cm[end], res.n_blownup, res.n_sigma, res.peak_Gamma)
    catch e
        e isa InterruptException && rethrow()
        global n_fail += 1
        @printf("       FAILED: %s\n", sprint(showerror, e))
        empty!(rows)
        push!(rows, csv_row(vcat(base, [0, NaN, NaN, NaN, NaN, 0])))
    end

    # Write each sample as soon as it finishes (hi-fi samples take hours)
    open(output_file, "a") do f
        foreach(line -> println(f, line), rows)
    end
end


# Summary
println("\n" * "=" ^ 72)
println("TIME-SERIES SWEEP COMPLETE")
nshards > 1 && @printf("  Shard:      %d of %d\n", shard, nshards)
@printf("  Ran:        %d\n", length(my_ids))
@printf("  Succeeded:  %d\n", n_success)
@printf("  Failed:     %d\n", n_fail)
@printf("  Output:     %s\n", abspath(output_file))
println("=" ^ 72)
