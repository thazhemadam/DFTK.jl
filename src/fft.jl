import FFTW

# returns the lengths of the bounding rectangle in reciprocal space
# that encloses the sphere of radius Gmax
function bounding_rectangle(lattice::AbstractMatrix{T}, Gmax; tol=sqrt(eps(T))) where {T}
    # If |B G| ≤ Gmax, then
    # |Gi| = |e_i^T B^-1 B G| ≤ |B^-T e_i| Gmax = |A_i| Gmax
    # with B the reciprocal lattice matrix, e_i the i-th canonical
    # basis vector and A_i the i-th column of the lattice matrix
    Glims = [norm(lattice[:, i]) / 2T(π) * Gmax for i in 1:3]

    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one G vector for 1D or 2D systems)
    Glims = [Glim == 0 ? 0 : ceil(Int, Glim .- tol) for Glim in Glims]
    Glims
end
function diameter(lattice)
    diam = zero(eltype(lattice))
    # brute force search
    for vec in Vec3.(Iterators.product(-1:1, -1:1, -1:1))
        diam = max(diam, norm(lattice*vec))
    end
    diam
end

const _smallprimes = [2, 3, 5]

@doc raw"""
Determine the minimal grid size for the cubic basis set to be able to
represent product of orbitals (with the default `supersampling=2`).

Optionally optimize the grid afterwards for the FFT procedure by
ensuring factorization into small primes.

The function will determine the smallest parallelepiped containing the wave vectors
 ``|G|^2/2 \leq E_\text{cut} ⋅ \text{supersampling}^2``.
For an exact representation of the density resulting from wave functions
represented in the spherical basis sets, `supersampling` should be at least `2`.
"""
function compute_fft_size(lattice::AbstractMatrix{T}, Ecut;
                          supersampling=2,
                          tol=sqrt(eps(T)),
                          ensure_smallprimes=true) where T
    Gmax = supersampling * sqrt(2 * Ecut)
    Glims = bounding_rectangle(lattice, Gmax; tol=tol)

    fft_size = Vec3(2 .* Glims .+ 1)
    # Optimize FFT grid size: Make sure the number factorises in small primes only
    if ensure_smallprimes
        fft_size = nextprod.(Ref(_smallprimes), fft_size)
    end
    fft_size
end
function compute_fft_size(model::Model, Ecut; kwargs...)
    compute_fft_size(model.lattice, Ecut; kwargs...)
end

# This uses a more precise and slower algorithm than the one above,
# simply enumerating all G vectors and seeing where their difference
# is. It needs the kpoints to do so.
# TODO This function is strange ... it should only depend on the kcoords
#      It should be merged with build_kpoints somehow
function compute_fft_size_precise(lattice::AbstractMatrix{T}, Ecut, kpoints;
                                  supersampling=2, ensure_smallprimes=true) where T
    recip_lattice = 2T(π)*pinv(lattice')  # pinv in case one of the dimension is trivial
    recip_diameter = diameter(recip_lattice)
    Glims = [0, 0, 0]
    # get the bounding rectangle that contains all G-G' vectors
    # (and therefore densities and potentials)
    # This handles the case `supersampling=2`
    for kpt in kpoints
        for G in G_vectors(kpt)
            if norm(recip_lattice * (G + kpt.coordinate)) ≤ sqrt(2Ecut) - recip_diameter
                # each of the 8 neighbors (in ∞-norm) also belongs to the grid
                # so we can safely skip the search knowing at least one of them
                # will have larger |G-Gp|.
                # Savings with this trick are surprisingly small :
                # for silicon, 50% at Ecut 30, 70% at Ecut 100
                continue
            end
            for Gp in G_vectors(kpt)
                for i = 1:3
                    @inbounds Glims[i] = max(Glims[i], abs(G[i] - Gp[i]))
                end
            end
        end
    end
    if supersampling != 2
        # no guarantees there, we just do our best to satisfy the
        # target supersampling ratio
        Glims = round.(Int, supersampling ./ 2 .* Glims)
    end

    fft_size = Vec3(2 .* Glims .+ 1)
    # Optimize FFT grid size: Make sure the number factorises in small primes only
    if ensure_smallprimes
        fft_size = nextprod.(Ref(_smallprimes), fft_size)
    end
    fft_size
end

# Main entry point from pwbasis. Uses the above functions to find out
# the correct fft_size according to user specification
function validate_or_compute_fft_size(model::Model{T}, fft_size, Ecut, supersampling,
                                      variational, optimize_fft_size, kcoords) where {T}
    # compute if not provided
    if fft_size === nothing
        @assert variational
        fft_size = compute_fft_size(model, Ecut; supersampling=supersampling)
        if optimize_fft_size
            # We build a temporary set of kpoints here
            # This gymnastics is because build_kpoints builds index
            # mapping from the k-point-specific basis to the global
            # basis and thus the fft_size needs to be final at kpoint
            # construction time
            fft_size = Tuple{Int, Int, Int}(fft_size)
            kpoints_temp = build_kpoints(model, fft_size, kcoords, Ecut;
                                         variational=variational)
            fft_size = compute_fft_size_precise(model.lattice, Ecut, kpoints_temp;
                                                supersampling=supersampling)
        end
    end

    # validate
    if variational
        max_E = sum(abs2, model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
        Ecut > max_E && @warn(
            "For a variational method, Ecut should be less than the maximal kinetic " *
            "energy the grid supports ($max_E)"
        )
    else
        # ensure no other options are set
        @assert supersampling == 2
        @assert !optimize_fft_size
    end

    # TODO generic FFT is kind of broken for some fft sizes
    #      ... temporary workaround, see more details in fft_generic.jl
    fft_size = next_working_fft_size.(T, fft_size)
    fft_size = Tuple{Int, Int, Int}(fft_size)
    fft_size
end


# For Float32 there are issues with aligned FFTW plans, so we
# fall back to unaligned FFTW plans (which are generally discouraged).
_fftw_flags(::Type{Float32}) = FFTW.MEASURE | FFTW.UNALIGNED
_fftw_flags(::Type{Float64}) = FFTW.MEASURE

"""
Plan a FFT of type `T` and size `fft_size`, spending some time on finding an
optimal algorithm. (Inplace, out-of-place) x (forward, backward) FFT plans are returned.
"""
function build_fft_plans(T::Union{Type{Float32}, Type{Float64}}, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)
    ipFFT = FFTW.plan_fft!(tmp, flags=_fftw_flags(T))
    opFFT = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    # backward by inverting and stripping off normalizations
    ipFFT, opFFT, inv(ipFFT).p, inv(opFFT).p
end


# TODO Some grid sizes are broken in the generic FFT implementation
# in FourierTransforms, for more details see fft_generic.jl
# This function is needed to provide a noop fallback for grid adjustment for
# for floating-point types natively supported by FFTW
next_working_fft_size(::Type{Float32}, size) = size
next_working_fft_size(::Type{Float64}, size) = size


#==============================================================================#
#
# ForwardDiff + FFTW zone
#
#==============================================================================#
import ForwardDiff
import AbstractFFTs

# original PR by mcabbott: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495
# modified version: https://github.com/niklasschmitz/ForwardDiff.jl/blob/nfs/fft/src/fft.jl

ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) =
    Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N

ForwardDiff.tagtype(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = T
ForwardDiff.tagtype(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = T

# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region; kwargs...)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region; kwargs...)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region; kwargs...)

    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        # TODO handle ForwardDiff.Dual scaling factors (perhaps lazy evaluation?)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:ForwardDiff.Dual}) = 
            (Y .= _apply_plan(p, X))
        
        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            (Y .= _apply_plan(p, X))
    end
end

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    T = ForwardDiff.tagtype(eltype(x))
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual{T}(real(val), map(real, parts)),
            ForwardDiff.Dual{T}(imag(val), map(imag, parts)),
        )
    end
end

###
### DFTK setup specific
###

next_working_fft_size(::Type{<:ForwardDiff.Dual}, size) = size

_fftw_flags(::Type{<:ForwardDiff.Dual}) = FFTW.MEASURE | FFTW.UNALIGNED

# *** COPIED from fft_generic.jl *** TODO refactor
# A dummy wrapper around an out-of-place FFT plan to make it appear in-place
# This is needed for some generic FFT implementations, which do not have in-place plans
struct DummyInplace{opFFT}
    fft::opFFT
end
LinearAlgebra.mul!(Y, p::DummyInplace, X) = (Y .= mul!(similar(X), p.fft, X))
LinearAlgebra.ldiv!(Y, p::DummyInplace, X) = (Y .= ldiv!(similar(X), p.fft, X))

import Base: *, \, length
*(p::DummyInplace, X) = p.fft * X
\(p::DummyInplace, X) = p.fft \ X
length(p::DummyInplace) = length(p.fft)

function build_fft_plans(T::Type{<:Union{ForwardDiff.Dual,Complex{<:ForwardDiff.Dual}}}, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)
    opFFT  = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    opBFFT = FFTW.plan_bfft(tmp, flags=_fftw_flags(T))

    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)
    # backward by inverting and stripping off normalizations
    ipFFT, opFFT, ipBFFT, opBFFT
end

