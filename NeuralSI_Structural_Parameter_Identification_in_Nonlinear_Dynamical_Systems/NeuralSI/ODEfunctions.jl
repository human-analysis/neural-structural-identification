
using DifferentialEquations, CUDA, DiffEqFlux, ModelingToolkit
using LinearAlgebra, Statistics, Random, DelimitedFiles, BandedMatrices, Plots

function myforce(t)
    (-sign.(t .- forced_time) .+ 1).*force_magnitude/2
end

function beam_nonlinear!(du, u, p, t)
    du[1,:] = u[2,:]
    du[2,:] = -EI/rhoA.*(
                A2*p[1:Nx] .* A2*u[1,:]    .+ 
                A1*p[1:Nx] .* A3*u[1,:].*2 .+
                   p[1:Nx] .* A4*u[1,:] 
                ) .+ 
                myforce(t)./rhoA.*b .-
                p[Nx+1:end].*u[2,:]./rhoA.*b
end

function solveODE(func!, p=p, x0=x0, solver=solver, tspan=tspan, tl=tl)
    return solve(ODEProblem(func!, x0, tspan, p), solver, saveat = tl,)[1,:,:]
end

function get_BandedMatrix(N,dx)
    A4 = Array(BandedMatrix(
            -2 => ones(N-2)*(1),
            -1 => ones(N-1)*(-4),
            0  => ones(N)*(6),
            1  => ones(N-1)*(-4),
            2  => ones(N-2)*(1)))
    A4[1,1:3] = [1 -4/5 1/5]; A4[end,end-2:end] = reverse([1 -4/5 1/5]);

    A3 = Array(BandedMatrix(
        -2 => ones(N-2)*(-1/2),
        -1 => ones(N-1)*(1),
        0  => ones(N)*(0),
        1  => ones(N-1)*(-1),
        2  => ones(N-2)*(1/2)))
    A3[1,:] .= 0; A3[end,:] .= 0;

    A2 = Array(BandedMatrix(
        -1 => ones(N-1)*(1.0),
        0  => ones(N)*(-2.0),
        1  => ones(N-1)*(1.0)));
    A2[1,:] .= 0; A2[end,:] .= 0;

    A1 = Array(BandedMatrix(
        -1 => ones(N-1)*(-1/2),
        0  => ones(N)*(0),
        1  => ones(N-1)*(1/2)));
    A1[1,:] .= 0; A1[end,:] .= 0;
    
    return A1/dx, A2/dx^2, A3/dx^3, A4/dx^4
end

# function jetmap(data; tl=tl, title="")
#     # heatmap(xl[2:end-1], tl, Array(data)'.*1e3, c=:jet, 
#     #         xlabel="Space /(m)", ylabel="Time /(s)", title=title)
#     heatmap(Array(data)', c=:jet, title=title)
# end

function results(v; position_p=:bottomleft, position_c=:bottomright)
    up = visual_vp(v,position=position_p)
    uc = visual_vc(v,position=position_c)
    predict = solveODE(func!, v, x0, solver, tspan, tl)
    diff_u = abs.(y-predict)
    u1 = jetmap(y, "True")
    u2 = jetmap(predict, "Predict")
    u3 = jetmap(diff_u, "Error")
    u4 = plot()
    display(plot(up,uc,u1,u2,u3,u4,layout=(3,2),size=(600,600)))
end

function jetmap(data, title = "", tl=tl)
    heatmap(Array(data)'.*1e3, c=:jet, 
            xlabel="Space /(m)", ylabel="Time /(s)", title=title)
end