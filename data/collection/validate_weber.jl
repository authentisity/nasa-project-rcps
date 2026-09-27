# Validation of wing_sim.jl against Weber & Brebner (1951), "Low-Speed Tests on
# 45-deg Swept-Back Wings, Part I", ARC R&M 2882: untapered, untwisted wing
# with 45 deg sweep, aspect ratio 5, RAE 101 section, Re_c = 1.7e6. Same case
# as FLOWUnsteady's wing example; experimental CL/CD from Table 4B.
#
# Usage (angles of attack in degrees; default 4.2):
#   julia -t 16 --project=. validate_weber.jl [AOA ...]
#   FIDELITY=low julia -t 16 --project=. validate_weber.jl 4.2
#
# Appends one line per AOA to weber_validation.csv in the working directory.

using Printf
include(joinpath(@__DIR__, "wing_sim.jl"))

const EXP = Dict(2.1  => (CL=0.121, CD=NaN),
                 4.2  => (CL=0.238, CD=0.005),
                 6.3  => (CL=0.350, CD=0.012),
                 8.4  => (CL=0.456, CD=0.022),
                 10.5 => (CL=0.559, CD=0.035))

fidelity = get(ENV, "FIDELITY", "high")
nsteps   = parse(Int, get(ENV, "NSTEPS", "200"))
AOAs     = isempty(ARGS) ? [4.2] : parse.(Float64, ARGS)
out_file = "weber_validation.csv"

isfile(out_file) || open(f -> println(f, "fidelity,nsteps,AOA,CL,CD,Cm,CL_exp,CD_exp,CL_err,CD_err,wall_s"), out_file, "w")

for AOA in AOAs
    wall = @elapsed res = run_wing(; AOA, ar=5.0, tr=1.0, lambda=45.0, gamma=0.0,
                                     twist_tip=0.0, magVinf=49.7, rho=0.93,
                                     fidelity, nsteps,
                                     airfoil_polar="xf-rae101-il-1000000.csv",
                                     add_skinfriction=false)
    # Steady state: average over the last 10% of the steps
    k  = max(1, length(res.CL) - nsteps ÷ 10 + 1)
    CL = sum(res.CL[k:end]) / length(res.CL[k:end])
    CD = sum(res.CD[k:end]) / length(res.CD[k:end])
    Cm = sum(res.Cm[k:end]) / length(res.Cm[k:end])
    ex = get(EXP, AOA, (CL=NaN, CD=NaN))
    @printf("\nWeber & Brebner AOA=%.1f (%s, %d steps, %.0f s): CL=%.4f (exp %.3f, %+.1f%%)  CD=%.5f (exp %.3f)  Cm=%.4f\n",
            AOA, fidelity, nsteps, wall, CL, ex.CL, 100*(CL/ex.CL - 1), CD, ex.CD, Cm)
    flush(stdout)   # stdout redirected to a file is block-buffered
    open(out_file, "a") do f
        println(f, join([fidelity, nsteps, AOA, CL, CD, Cm, ex.CL, ex.CD,
                         CL/ex.CL - 1, CD/ex.CD - 1, wall], ","))
    end
end
