# Isolated-wing FLOWUnsteady simulation shared by the sweep and validation
# scripts. `run_wing` returns the per-step CL, CD and Cm transient.
#
# Fidelity presets:
#   "high"  Settings of the high-fidelity preset in FLOWUnsteady's PROWIM
#           example, which replicates Alvarez & Ning 2023, J. Aircraft,
#           doi:10.2514/1.C037279 (an untapered, unswept wing with two
#           propellers). The rVPM and SFS model are from Alvarez & Ning 2023,
#           AIAA J., doi:10.2514/1.J063045. Actuator surface model
#           (vortex sheet), dynamic SFS LES model, RK3, 5 sheds per step,
#           lambda = 2.125, 100 elements per semi-span (loads converged to <1%
#           for n >= 100, Alvarez 2022 dissertation, wing convergence study).
#   "low"   Settings of FLOWUnsteady's Weber & Brebner wing example (actuator
#           line model, no SFS model). Only meant for smoke tests.
#
# Reference quantities: CL and CD are normalized by the projected planform area
# S = b (c_root + c_tip) / 2, and Cm by S times the mean aerodynamic chord about
# the quarter chord of the MAC (nose-up positive).

import FLOWUnsteady as uns
import FLOWUnsteady: vlm, vpm

# FLOWUnsteady 3.4 still passes `index` to add_particle when it builds the ASM
# vortex sheet of a wing, a keyword FLOWVPM 4 removed (the rotor path already
# dropped it). The index is only read by the "averaged"/"weighted" KJ force
# types, not the "regular" one used here, so discard it and forward the call.
function vpm.add_particle(pfield::vpm.ParticleField, X::AbstractVector,
                          Gamma::AbstractVector, sigma::Real; index=nothing, optargs...)
    return invoke(vpm.add_particle, Tuple{vpm.ParticleField, Any, Any, Any},
                  pfield, X, Gamma, sigma; optargs...)
end

"""
Planform quantities of `vlm.simpleWing(b, ar, tr, twist, lambda, gamma)`,
which places the root leading edge at the origin, sets c_tip = b/ar and
c_root = c_tip/tr, and sweeps (dihedrals) the leading edge by lambda (gamma).
"""
function planform(b, ar, tr, lambda, gamma)
    c_tip  = b / ar
    c_root = c_tip / tr
    S      = b * (c_root + c_tip) / 2
    mac    = 2/3 * c_root * (1 + tr + tr^2) / (1 + tr)
    y_mac  = b/6 * (1 + 2*tr) / (1 + tr)
    Xref   = [y_mac * tand(lambda) + mac/4, 0.0, y_mac * tand(gamma)]
    return (; c_tip, c_root, S, mac, Xref)
end

function fidelity_settings(fidelity, AOA)
    if fidelity == "high"
        return (n = 100, p_per_step = 5, lambda_vpm = 2.125, sigma_vlm_surf_b = 1/200,
                vlm_rlx = 0.3, shed_starting = AOA < 8, unsteady_shedcrit = 0.001,
                vortexsheet = true,
                vpm_SFS = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
                                         alpha=0.999, maxC=1.0,
                                         clippings=[vpm.clipping_backscatter]))
    elseif fidelity == "low"
        return (n = 50, p_per_step = 1, lambda_vpm = 2.0, sigma_vlm_surf_b = 0.05,
                vlm_rlx = 0.7, shed_starting = true, unsteady_shedcrit = 0.01,
                vortexsheet = false, vpm_SFS = vpm.SFS_none)
    else
        error("Unknown fidelity \"$fidelity\"; expected \"high\" or \"low\"")
    end
end

"""
    run_wing(; AOA, ar, tr, lambda, gamma, twist_tip, magVinf, ...)

Simulate an isolated simpleWing from rest until its wake is `wake_factor` spans
long. Returns `(; t, CL, CD, Cm, planform, n_removed, peak_Gamma)` where the
arrays hold one value per time step (steps 3..nsteps, as logged by
FLOWUnsteady's wing monitor), `n_removed` counts blown-up particles removed and
`peak_Gamma` is the largest kept particle strength over the removal bound.
"""
function run_wing(; AOA, ar, tr, lambda, gamma, twist_tip, magVinf,
                    b = 2.489, rho = 1.225, twist_root = 0.0,
                    fidelity = "high", nsteps = 200, wake_factor = 2.75,
                    r_expansion = 10.0,
                    airfoil_polar = "xf-n0012-il-500000-n5.csv",
                    add_skinfriction = true, thickness = 0.12, v_lvl = 1,
                    verbose_nsteps = nsteps)

    fs = fidelity_settings(fidelity, AOA)
    pf = planform(b, ar, tr, lambda, gamma)
    qinf = 0.5 * rho * magVinf^2

    Vinf(X, t) = magVinf * [cosd(AOA), 0.0, sind(AOA)]

    wing = vlm.simpleWing(b, ar, tr, twist_root, lambda, gamma;
                          twist_tip=twist_tip, n=fs.n, r=r_expansion, central=false)
    system = vlm.WingSystem()
    vlm.addwing(system, "Wing", wing)
    vehicle = uns.VLMVehicle(system; vlm_system=system, wake_system=system)

    Vvehicle(t)     = zeros(3)
    anglevehicle(t) = zeros(3)
    maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

    ttot = wake_factor * b / magVinf
    dt   = ttot / nsteps
    simulation = uns.Simulation(vehicle, maneuver, 0.0, 0.0, ttot;
                                Vinit=zeros(3), Winit=zeros(3))

    sigma_vpm_overwrite = fs.lambda_vpm * magVinf * dt / fs.p_per_step
    sigma_vlm_surf      = fs.sigma_vlm_surf_b * b
    sigma_tbv           = fs.vortexsheet ? thickness * pf.c_tip / 128 : nothing

    m = vlm.get_m(system)
    max_particles = (nsteps + 1) * (m * (fs.p_per_step + 1) + fs.p_per_step)
    max_static_particles = nothing
    if fs.vortexsheet
        # The vortex sheet's static particles share the main field during each
        # step (FLOWUnsteady's _static_particles): per horseshoe, 2.125 c/sigma
        # on each trailing bound vortex and on the lifting sheet. Bounded here
        # with the root chord; a fixed budget (PROWIM's 10^6, sized for its
        # rectangular wing) overflows for tr < 0.8, since sigma_tbv ~ c_tip.
        np_chord(sigma) = ceil(Int, 2.125 * pf.c_root / sigma) + 1
        max_static_particles = m * (2 * np_chord(sigma_tbv) + np_chord(sigma_vlm_surf))
        max_particles += max_static_particles
    end

    # Kutta-Joukowski force at the midpoint of each lifting bound vortex plus
    # parasitic drag from the airfoil polar (the vortex sheet only changes the
    # VLM-on-VPM coupling, not this force, with the "regular" KJ force type)
    calc_aerodynamicforce_fun = uns.generate_calc_aerodynamicforce(;
                                    add_parasiticdrag=true,
                                    add_skinfriction=add_skinfriction,
                                    airfoilpolar=airfoil_polar)

    Dhat = [cosd(AOA), 0.0, sind(AOA)]
    Shat = [0.0, 1.0, 0.0]
    Lhat = uns.cross(Dhat, Shat)

    # The monitor normalizes by qinf*b^2/ar_ref, so ar_ref = b^2/S gives
    # planform-area coefficients
    cl_out, cd_out = Float64[], Float64[]
    wing_monitor = uns.generate_monitor_wing(wing, Vinf, b, b^2 / pf.S,
                                             rho, qinf, nsteps;
                                             calc_aerodynamicforce_fun=calc_aerodynamicforce_fun,
                                             L_dir=Lhat, D_dir=Dhat,
                                             out_CLwing=cl_out, out_CDwing=cd_out,
                                             save_path=nothing, disp_plot=false)

    t_hist, cm_hist = Float64[], Float64[]

    # Blown-up particle guard. A particle whose strength diverges (or whose
    # rVPM core size is driven to <= 0) crashes the next FMM call: FLOWVPM's
    # regularization autotuning cannot bracket its root. PROWIM's lower
    # fidelity presets remove particles stronger than 10x a CL = 2 bound vortex
    # shed over one substep; the same upper bound is applied here, with c_root,
    # but not their lower bound, so runs that do not blow up are unchanged.
    # (The strongest particle over the first 8 steps of sweep sample 3,
    # starting vortex included, is 0.10 of it.) Called after the step, when
    # only free particles are in the field.
    Gamma_max = 10 * 2.0 * magVinf * pf.c_root / 2 * magVinf * dt / fs.p_per_step
    n_removed, peak_Gamma = Ref(0), Ref(0.0)
    function remove_blownup(PFIELD)
        n = 0
        for i in vpm.get_np(PFIELD):-1:1
            G, sigma = vpm.get_Gamma(PFIELD, i), vpm.get_sigma(PFIELD, i)[]
            G2 = G[1]^2 + G[2]^2 + G[3]^2
            if G2 <= Gamma_max^2 && sigma > 0          # false for NaN
                peak_Gamma[] = max(peak_Gamma[], sqrt(G2))
            else
                vpm.remove_particle(PFIELD, i)
                n += 1
            end
        end
        n > 0 && println("\t\tstep $(PFIELD.nt): removed $n blown-up particles")
        n_removed[] += n
    end

    # The wing monitor stores each element's force in wing.sol["Ftot"]; the
    # force acts at the midpoint of the element's lifting bound vortex A-B
    function monitor(sim, PFIELD, T, DT; optargs...)
        ret = wing_monitor(sim, PFIELD, T, DT; optargs...)
        remove_blownup(PFIELD)
        if PFIELD.nt > 2
            M = zeros(3)
            for (i, F) in enumerate(wing.sol["Ftot"])
                HS = vlm.getHorseshoe(wing, i)
                X  = (HS[2] + HS[3]) / 2
                M .+= uns.cross(X - pf.Xref, F)
            end
            push!(t_hist, PFIELD.t)
            push!(cm_hist, uns.dot(M, Shat) / (qinf * pf.S * pf.mac))
        end
        return ret
    end

    uns.run_simulation(simulation, nsteps;
                       Vinf=Vinf,
                       rho=rho,
                       p_per_step=fs.p_per_step,
                       max_particles=max_particles,
                       max_static_particles=max_static_particles,
                       vpm_integration=vpm.rungekutta3,
                       vpm_SFS=fs.vpm_SFS,
                       sigma_vlm_solver=-1,
                       sigma_vlm_surf=sigma_vlm_surf,
                       sigma_rotor_surf=sigma_vlm_surf,
                       sigma_vpm_overwrite=sigma_vpm_overwrite,
                       vlm_vortexsheet=fs.vortexsheet,
                       vlm_vortexsheet_overlap=2.125,
                       vlm_vortexsheet_distribution=uns.g_pressure,
                       vlm_vortexsheet_sigma_tbv=sigma_tbv,
                       vlm_rlx=fs.vlm_rlx,
                       shed_unsteady=true,
                       shed_starting=fs.shed_starting,
                       unsteady_shedcrit=fs.unsteady_shedcrit,
                       extra_runtime_function=monitor,
                       save_path=nothing,
                       v_lvl=v_lvl,
                       verbose_nsteps=verbose_nsteps)

    return (; t=t_hist, CL=copy(cl_out), CD=copy(cd_out), Cm=cm_hist, planform=pf,
              n_removed=n_removed[], peak_Gamma=peak_Gamma[] / Gamma_max)
end
