## In the package as examples/ex1/doubleintegrator1D.jl 
#import import Pkg; Pkg.add("Package_Name") OR Pkg.add(url="https:// github_link.com")


using L1DRAC
using CUDA
using LinearAlgebra
using Distributions
using ControlSystemsBase
using StaticArrays
using Plots
using JLD2

testo = zeros(1,12)
for i in axes(testo)
    println(i)
end




#=
struct Location
    name::String
    lat::Float32
    lon::Float32
end

loc1 = Location("Los Angeles", 34.0522,-118.2437)

loc1.name   # "Los Angeles"
loc1.lat    # 34.0522
loc1.lon    # -118.2437

sites = Location[]
push!(sites, Location("Los Angeles", 34.0522,-118.2437))
push!(sites, Location("Las Vegas", 36.1699,-115.1398))


A = @SVector [0.0; 1.0]
println(transpose(A)*A)

if 6 == 6
    if 6 == 6
        error("reuse_existing_connection=True but no PyBullet connection is active.")
    end
else
    #physics_client = #EDITME_connect(connection_mode)
end

@__DIR__
SMatrix{2,2}(1.0I)
teststruct = sys_dims(1,2,3)
fieldnames(typeof(teststruct))

Matrix(I, 12, 6)

Δₜ = 1e-4
A = Matrix{Float32}(I, 12, 12);
A[1, 4] = Δₜ;
A[2, 5] = Δₜ;
A[3, 6] = Δₜ;
A[7, 10] = Δₜ;
A[8, 11] = Δₜ;
A[9, 12] = Δₜ;
A

A[4, 6] = -g * s_yaw * Δₜ
A[4, 7] =  g * c_yaw * Δₜ
A[5, 7] = -g * c_yaw * Δₜ
A[5, 8] = -g * s_yaw * Δₜ
#Δₜ
#@SMatrix [0.0; 1.0]
=#

#=
Ntraj = 10

tspan = (0.0, 5.0)
Δₜ = 1e-4 # Time step size
Δ_saveat = 1e2 * Δₜ # Needs to be an integer multiple of Δₜ
simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

# System Dimensions
n, m, d = 2, 1, 2
system_dimensions = sys_dims(n, m, d)

# Double integrator dynamics
A = @SMatrix [0.0 1.0; 0.0 0.0]
B = @SMatrix [0.0; 1.0]

# Baseline controller via pole placement
λ = 10.0 # Stability margin
sys = ss(A, B, SMatrix{2,2}(1.0I), 0.0)
K = SMatrix{1,2}(place(sys, -λ * ones(2)))
dp = (; K) # Dynamics params for GPU

lol = 2
lol2 = 3
=#

