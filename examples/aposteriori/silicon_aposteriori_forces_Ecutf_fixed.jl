# Computation of error estimate and corrections for the forces for the linear
# silicon system, in the form Ax=b
#
# Very basic setup, useful for testing
using DFTK
import DFTK: apply_K, apply_Ω, solve_ΩplusK, compute_projected_gradient
import DFTK: proj_tangent, proj_tangent!, proj_tangent_kpt
using HDF5
using PyPlot

include("aposteriori_forces.jl")
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8 + [0.42, 0.35, 0.24] ./ 20, -ones(3)/8]]

#  model = Model(lattice; atoms=atoms, terms=[Kinetic(), AtomicLocal()])
model = model_LDA(lattice, atoms)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 60   # kinetic energy cutoff in Hartree
tol = 1e-10
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)

filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

## reference values
φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end
f_ref = compute_forces(scfres_ref)

## min and max Ecuts for the two grid solution
Ecut_min = 5
Ecut_max = 50

Ecut_list = Ecut_min:5:Ecut_max
K = length(Ecut_list)
diff_list = zeros((K,K))
diff_list_res = zeros((K,K))
diff_list_schur = zeros((K,K))
Mres_list = zeros(K)
Mschur_list = zeros(K)
Merr_list = zeros(K)
err_list = zeros(K)
res_list = zeros(K)

i = 0
j = 0

for Ecut_g in Ecut_list

    println("---------------------------")
    println("Ecut grossier = $(Ecut_g)")
    global i,j
    i += 1
    j = i
    basis_g = PlaneWaveBasis(model, Ecut_g; kgrid=kgrid)

    ## solve eigenvalue system
    scfres_g = self_consistent_field(basis_g, tol=tol,
                                     determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                     is_converged=DFTK.ScfConvergenceDensity(tol))
    ham_g = scfres_g.ham
    ρ_g = scfres_g.ρ

    ## quantities
    φ = similar(scfres_g.ψ)
    for ik = 1:Nk
        φ[ik] = scfres_g.ψ[ik][:,1:N]
    end
    f_g = compute_forces(scfres_g)

    for Ecut_f in [Ecut_ref]

        println("Ecut fin = $(Ecut_f)")
        # fine grid
        basis_f = PlaneWaveBasis(model, Ecut_f; kgrid=kgrid)

        # compute residual and keep only LF
        φr = DFTK.transfer_blochwave(φ, basis_g, basis_f)
        res = compute_projected_gradient(basis_f, φr, occupation)
        resLF = DFTK.transfer_blochwave(res, basis_f, basis_g)

        # compute hamiltonian
        ρr = compute_density(basis_f, φr, occupation)
        _, ham_f = energy_hamiltonian(basis_f, φr, occupation; ρ=ρr)

        ## prepare P
        kpt = basis_f.kpoints[1]
        P = [PreconditionerTPA(basis_f, kpt) for kpt in basis_f.kpoints]
        for ik = 1:length(P)
            DFTK.precondprep!(P[ik], φr[ik])
        end

        ## compute error
        err = compute_error(basis_f, φr, φ_ref)
        Merr = apply_metric(φr, P, err, apply_sqrt_M)

        ## Rayleigh coefficients
        Λ = map(enumerate(φr)) do (ik, ψk)
            Hk = ham_f.blocks[ik]
            Hψk = Hk * ψk
            ψk'Hψk
        end

        resHF = res - DFTK.transfer_blochwave(resLF, basis_g, basis_f)
        resHF = apply_metric(φr, P, resHF, apply_inv_T)
        ΩpKres = apply_Ω(resHF, φr, ham_f, Λ) .+ apply_K(basis_f, resHF, φr, ρr, occupation)
        ΩpKresLF = DFTK.transfer_blochwave(ΩpKres, basis_f, basis_g)
        rhs = resLF - ΩpKresLF
        eLF = solve_ΩplusK(basis_g, φ, rhs, occupation)
        e = DFTK.transfer_blochwave(eLF, basis_g, basis_f)

        # Apply M^+-1/2
        Me = apply_metric(φr, P, e, apply_sqrt_M)
        Mres = apply_metric(φr, P, res, apply_inv_sqrt_M)
        Mschur = [Mres[1] + Me[1]]

        ##  plot carots
        #  G_energies = DFTK.G_vectors_cart(basis_f.kpoints[1])
        #  normG = norm.(G_energies)
        #  figure(i)
        #  title("Ecut_g = $(Ecut_g)")
        #  plot(Merr[1][sortperm(normG)], "r", label="Merr")
        #  plot(Mschur[1][sortperm(normG)], "g", label="Mres_schur")
        #  plot(Mres[1][sortperm(normG)], "b", label="Mres")
        #  xlabel("index of G by increasing norm")
        #  legend()

        #  figure(10+i)
        #  plot(res[1][sortperm(normG)], "b", label="res")
        #  plot(err[1][sortperm(normG)], "r", label="err")
        #  xlabel("index of G by increasing norm")
        #  legend()

        # approximate forces f-f*
        f_res = compute_forces_estimate(basis_f, Mres, φr, P, occupation)
        f_schur = compute_forces_estimate(basis_f, Mschur, φr, P, occupation)

        diff_list[i,j] = abs(f_g[1][2][1]-f_ref[1][2][1])
        diff_list_res[i,j] = abs(f_res[1][2][1])
        diff_list_schur[i,j] = abs(f_schur[1][2][1])
        Mres_list[i] = norm(Mres)
        Merr_list[i] = norm(Merr)
        res_list[i] = norm(res)
        err_list[i] = norm(err)
        Mschur_list[i] = norm(Mschur)
        j += 1
    end
end

h5open("Ecutf_fixed_forces.h5", "w") do file
    println("writing h5 file")
    file["Ecut_ref"] = Ecut_ref
    file["Ecut_list"] = collect(Ecut_list)
    file["diff_list"] = diff_list
    file["diff_list_res"] = diff_list_res
    file["diff_list_schur"] = diff_list_schur
    file["Merr_list"] = Merr_list
    file["Mres_list"] = Mres_list
    file["Mschur_list"] = Mschur_list
end

figure()
semilogy(Ecut_list, [diff_list[i,i] for i in 1:length(Ecut_list)], label="F-F*")
semilogy(Ecut_list, [diff_list_res[i,i] for i in 1:length(Ecut_list)], label="Fres")
semilogy(Ecut_list, [diff_list_schur[i,i] for i in 1:length(Ecut_list)], label="Fschur")
legend()

figure()
semilogy(Ecut_list, Merr_list, label="Merr")
semilogy(Ecut_list, Mres_list, label="Mres")
semilogy(Ecut_list, err_list, label="err")
semilogy(Ecut_list, res_list, label="res")
semilogy(Ecut_list, Mschur_list, label="Mschur")
legend()
