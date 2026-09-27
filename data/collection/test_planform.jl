# Checks `planform` (wing_sim.jl) against the discretized simpleWing geometry:
# area, mean aerodynamic chord and its leading-edge location, integrated
# element-by-element from FLOWVLM's leading/trailing edges.
#
# Usage:
#   julia --project=. test_planform.jl

using Test
include(joinpath(@__DIR__, "wing_sim.jl"))

function integrated_planform(wing)
    S = Sc2 = Scy = Sxle = Szle = 0.0
    for i in 1:vlm.get_m(wing)
        LE1, LE2 = vlm.getLE(wing, i), vlm.getLE(wing, i + 1)
        TE1, TE2 = vlm.getTE(wing, i), vlm.getTE(wing, i + 1)
        c1, c2 = TE1[1] - LE1[1], TE2[1] - LE2[1]
        dy = abs(LE2[2] - LE1[2])
        # Chord and LE are linear in y within each element (Simpson's rule is exact)
        cm, xm, zm = (c1 + c2)/2, (LE1[1] + LE2[1])/2, (LE1[3] + LE2[3])/2
        ym = abs(LE1[2] + LE2[2])/2
        simpson(f1, fm, f2) = dy * (f1 + 4fm + f2) / 6
        S    += simpson(c1, cm, c2)
        Sc2  += simpson(c1^2, cm^2, c2^2)
        Scy  += simpson(c1*abs(LE1[2]), cm*ym, c2*abs(LE2[2]))
        Sxle += simpson(c1*LE1[1], cm*xm, c2*LE2[1])
        Szle += simpson(c1*LE1[3], cm*zm, c2*LE2[3])
    end
    # MAC, and the chord-weighted spanwise station/LE point (the MAC's location)
    return (S=S, mac=Sc2/S, y_mac=Scy/S, x_le=Sxle/S, z_le=Szle/S)
end

@testset "planform" begin
    b = 2.489
    for (ar, tr, lambda, gamma) in [(5.0, 1.0, 45.0, 0.0), (3.0, 0.3, 0.0, 10.0),
                                    (10.0, 0.6, 30.0, -5.0), (7.0, 0.45, 50.0, 4.0)]
        wing = vlm.simpleWing(b, ar, tr, 0.0, lambda, gamma; n=40, r=3.0)
        ref  = integrated_planform(wing)
        pf   = planform(b, ar, tr, lambda, gamma)
        @test pf.S ≈ ref.S rtol=1e-10
        @test pf.mac ≈ ref.mac rtol=1e-10
        # For a trapezoidal wing, the chord-weighted LE point is the MAC's LE
        @test pf.Xref[1] - pf.mac/4 ≈ ref.x_le rtol=1e-10
        @test pf.Xref[3] ≈ ref.z_le atol=1e-12
    end
end
