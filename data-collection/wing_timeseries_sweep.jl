# Parametric sweep over wing designs that logs the full per-timestep transient
# of each unsteady VLM+VPM simulation as the wake develops from t=0 to steady
# state, producing time-series training data for a recurrent dynamics model.
#
# Output: wing_timeseries_data.csv, long format (one row per sample+timestep).
#
# Usage:
#   julia -t auto --project=. wing_timeseries_sweep.jl
#
# Optional sharding across processes: shard k of N runs samples k, k+N, k+2N,
# ... and writes its own CSV. Concatenate the shard files after all finish.
#   SHARD=2 NSHARDS=8 julia -t 4 --project=. wing_timeseries_sweep.jl

import FLOWUnsteady as uns
import FLOWVLM as vlm
using Random
using Printf

# Sweep configuration
num_samples     = 500

AOA_range       = (0.0, 12.0)       # (deg) angle of attack
ar_range        = (3.0, 10.0)       # Aspect ratio (b / c_tip)
tr_range        = (0.3, 1.0)        # Taper ratio (c_tip / c_root)
lambda_range    = (0.0, 50.0)       # (deg) sweep angle
gamma_range     = (-5.0, 10.0)      # (deg) dihedral angle
twist_tip_range = (-5.0, 5.0)       # (deg) tip twist (washout if negative)
magVinf_range   = (20.0, 80.0)      # (m/s) freestream velocity

# Fixed parameters
b               = 2.489             # (m) span length
rho             = 1.225             # (kg/m^3) air density (sea level ISA)
twist_root      = 0.0               # (deg) root twist (fixed at zero)
mu              = 1.81e-5           # (Pa*s) dynamic viscosity of air at ~15C

# Solver resolution
n_elem          = 50                # Spanwise VLM elements per side
r_expansion     = 10.0              # Geometric expansion ratio of elements
nsteps          = 200               # Number of time steps (== trajectory length)
p_per_step      = 1                 # Particle sheds per time step
lambda_vpm      = 2.0               # VPM core overlap factor
vlm_rlx         = 0.7               # VLM relaxation factor
wake_factor     = 2.75              # Wake length as multiple of span

airfoil_polar   = "xf-n0012-il-500000-n5.csv"

# Execution / output
nshards         = parse(Int, get(ENV, "NSHARDS", "1"))
shard           = parse(Int, get(ENV, "SHARD", "1"))
@assert 1 <= shard <= nshards "SHARD must be in 1:NSHARDS"

output_file     = nshards == 1 ? "wing_timeseries_data.csv" :
                  "wing_timeseries_data_shard$(shard)of$(nshards).csv"
checkpoint_every = 10
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

csv_row(vals) = join([@sprintf("%.6g", v) for v in vals], ",")

function flush_rows!(buffer, file)
    isempty(buffer) && return
    open(file, "a") do f
        foreach(line -> println(f, line), buffer)
    end
    empty!(buffer)
end


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

println("Total configurations to run: $num_samples\n")

calc_aerodynamicforce_fun = uns.generate_calc_aerodynamicforce(;
                                add_parasiticdrag=true,
                                add_skinfriction=true,
                                airfoilpolar=airfoil_polar
                                )

const CSV_HEADER = join([
    "sample_id",
    "AOA", "ar", "tr", "lambda", "gamma", "twist_tip", "magVinf",
    "qinf", "S_ref", "c_root", "c_tip", "Re_approx",
    "step", "t", "CL", "CD", "Cm",
    "converged"
], ",")

open(output_file, "w") do f
    println(f, CSV_HEADER)
end

csv_buffer = String[]


# Run the sweep, logging the full transient of each sample
println("=" ^ 72)
println("STARTING TIME-SERIES SWEEP  ($num_samples configurations)")
nshards > 1 && println("Shard $shard of $nshards  →  $output_file")
println("=" ^ 72)

n_success = 0
n_fail    = 0

for (idx, cfg) in enumerate(configs)

    # Not this shard's sample
    if mod1(idx, nshards) != shard
        continue
    end

    @printf("\n[%d/%d] AOA=%5.1f°  ar=%4.1f  tr=%.2f  Λ=%5.1f°  Γ=%5.1f°  twist=%5.1f°  V=%5.1f m/s\n",
            idx, num_samples,
            cfg.AOA, cfg.ar, cfg.tr, cfg.lambda, cfg.gamma, cfg.twist_tip, cfg.magVinf)

    # Per-step trajectory buffers (filled by the monitor wrapper below)
    t_hist  = Float64[]
    cl_hist = Float64[]
    cd_hist = Float64[]
    cm_hist = Float64[]

    qinf = S_ref = c_root = c_tip = Re_approx = NaN
    converged = false

    try
        # Derived geometry
        c_tip      = b / cfg.ar
        c_root     = c_tip / cfg.tr
        mean_chord = (c_root + c_tip) / 2.0
        S_ref      = b^2 / cfg.ar
        qinf       = 0.5 * rho * cfg.magVinf^2
        Re_approx  = rho * cfg.magVinf * mean_chord / mu

        Vinf(X, t) = cfg.magVinf * [cosd(cfg.AOA), 0.0, sind(cfg.AOA)]

        # Geometry / vehicle
        wing = vlm.simpleWing(b, cfg.ar, cfg.tr, twist_root, cfg.lambda, cfg.gamma;
                              twist_tip=cfg.twist_tip, n=n_elem, r=r_expansion, central=false)
        system = vlm.WingSystem()
        vlm.addwing(system, "Wing", wing)
        vehicle = uns.VLMVehicle(system; vlm_system=system, wake_system=system)

        Vvehicle(t)     = zeros(3)
        anglevehicle(t) = zeros(3)
        maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

        wakelength = wake_factor * b
        ttot       = wakelength / cfg.magVinf
        dt         = ttot / nsteps
        max_particles = (nsteps + 1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step + 1) + p_per_step)

        simulation = uns.Simulation(vehicle, maneuver, 0.0, 0.0, ttot;
                                    Vinit=zeros(3), Winit=zeros(3))

        sigma_vpm_overwrite = lambda_vpm * cfg.magVinf * dt / p_per_step
        sigma_vlm_surf      = 0.05 * b

        Dhat = [cosd(cfg.AOA), 0.0, sind(cfg.AOA)]
        Lhat = uns.cross(Dhat, [0, 1, 0])
        Shat = [0, 1, 0]
        c_ref = b / cfg.ar
        Xac = [0.25 * c_root, 0.0, 0.0]

        # The wing monitor appends one CL/CD value per step (for nt>2) into these
        cl_out = Float64[]
        cd_out = Float64[]

        wing_monitor = uns.generate_monitor_wing(wing, Vinf, b, cfg.ar,
                                            rho, qinf, nsteps;
                                            calc_aerodynamicforce_fun=calc_aerodynamicforce_fun,
                                            L_dir=Lhat,
                                            D_dir=Dhat,
                                            out_CLwing=cl_out,
                                            out_CDwing=cd_out,
                                            save_path=nothing,
                                            disp_plot=false)

        # Run the wing monitor (updates wing.sol and pushes CL/CD), then derive
        # Cm from the current solution
        function ts_monitor(sim, PFIELD, T, DT; optargs...)
            ret = wing_monitor(sim, PFIELD, T, DT; optargs...)
            if PFIELD.nt > 2
                Xs = [vlm.getControlPoint(wing, i) for i in 1:vlm.get_m(wing)]
                Fs = wing.sol["Ftot"]
                M  = sum(uns.cross(X - Xac, F) for (X, F) in zip(Xs, Fs))
                Cm_t = uns.dot(M, Shat) / (qinf * S_ref * c_ref)
                push!(t_hist,  PFIELD.t)
                push!(cm_hist, Cm_t)
                push!(cl_hist, cl_out[end])
                push!(cd_hist, cd_out[end])
            end
            return ret
        end

        uns.run_simulation(simulation, nsteps;
                           Vinf=Vinf,
                           rho=rho,
                           p_per_step=p_per_step,
                           max_particles=max_particles,
                           sigma_vlm_solver=-1,
                           sigma_vlm_surf=sigma_vlm_surf,
                           sigma_rotor_surf=sigma_vlm_surf,
                           sigma_vpm_overwrite=sigma_vpm_overwrite,
                           shed_starting=true,
                           vlm_rlx=vlm_rlx,
                           vpm_integration=uns.vpm.rungekutta3,
                           extra_runtime_function=ts_monitor,
                           save_path=nothing,
                           v_lvl=1,
                           verbose_nsteps=nsteps)

        converged = length(cl_hist) > 0
        if converged
            global n_success += 1
            @printf("       logged %d steps  |  steady CL=%.4f  CD=%.5f  Cm=%.4f\n",
                    length(cl_hist), cl_hist[end], cd_hist[end], cm_hist[end])
        else
            global n_fail += 1
            @printf("       FAILED: no steps logged\n")
        end

    catch e
        global n_fail += 1
        @printf("       FAILED: %s\n", sprint(showerror, e))
    end

    # Append rows for this sample; a failed sample gets one placeholder row so
    # it is still accounted for
    base = [idx,
            cfg.AOA, cfg.ar, cfg.tr, cfg.lambda, cfg.gamma, cfg.twist_tip, cfg.magVinf,
            qinf, S_ref, c_root, c_tip, Re_approx]
    if converged
        nrec = min(length(t_hist), length(cl_hist), length(cd_hist), length(cm_hist))
        for k in 1:nrec
            push!(csv_buffer, csv_row(vcat(base, [k, t_hist[k], cl_hist[k], cd_hist[k], cm_hist[k], 1])))
        end
    else
        push!(csv_buffer, csv_row(vcat(base, [0, NaN, NaN, NaN, NaN, 0])))
    end

    # Checkpoint save
    if (n_success + n_fail) % checkpoint_every == 0 || idx == num_samples
        flush_rows!(csv_buffer, output_file)
        @printf("       [Checkpoint: %d/%d done — %d ok, %d failed]\n",
                idx, num_samples, n_success, n_fail)
    end
end

flush_rows!(csv_buffer, output_file)


# Summary
println("\n" * "=" ^ 72)
println("TIME-SERIES SWEEP COMPLETE")
if nshards > 1
    n_assigned = count(i -> mod1(i, nshards) == shard, 1:num_samples)
    @printf("  Shard:      %d of %d (%d samples)\n", shard, nshards, n_assigned)
end
@printf("  Total:      %d\n", num_samples)
@printf("  Succeeded:  %d\n", n_success)
@printf("  Failed:     %d\n", n_fail)
@printf("  Output:     %s\n", abspath(output_file))
println("=" ^ 72)
