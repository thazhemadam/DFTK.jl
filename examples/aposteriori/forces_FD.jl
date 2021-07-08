using ForwardDiff

function forces(basis, occupation, ψ, δρ)
    compute_forces(basis, ψ, occupation; ρ=δρ)
end

function δforces(basis, occupation, ψ, δψ)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    function f(ε)
        forces(basis, occupation, ψ .+ ε .* δψ, ε .* δρ)
    end
    ForwardDiff.derivative(f, 0)
end
