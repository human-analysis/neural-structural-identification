
include("ODEfunctions.jl")
include("Networks.jl")
include("FrechetDistance.jl")

# Space & Time
Nx = 16; Lx = 0.4f0; xl = LinRange(0,Lx,Nx+2); dx = xl[2] - xl[1]; xll = xl[2:end-1];
Nt = 160; tmax = 0.045f0; tl = LinRange(0,tmax,Nt); dt = tl[2] - tl[1]; tspan = (0,tmax);
A1, A2, A3, A4 = get_BandedMatrix(Nx,dx);
# Geometry & Modulus
thick = 0.005f0; width = 0.05f0; E = 70f9; rho = 2700f0; 
rhoA = rho*thick*width; I = width*thick^3/12; EI = E*I; 
vp0 = Array(.25*sin.(xll/Lx*2*pi).+ 1.0); vc0 = Array(0.8 .+ 0.3 .*xll./Lx).*20;
# Force
forced_time = 0.02f0; force_magnitude = 1000f0; 
# Initial condition & Solver
p0 = cat(vp0,vc0;dims=1); solver = RK4(); #ImplicitEulerExtrapolation();
x0 = zeros(Float32,2,Nx); b = ones(Nx,1); b[[1, Nx]]=[0 0];

# ------------------ ground truth ------------------
func! = beam_nonlinear!
@time y = solveODE(func!,p0);
jetmap(y)

# ------------------ def network -------------------
sample_ratio = 0.5f0; epochs = 20; minibatch_size = 16;
xlmap = repeat(xl./Lx, 1, Nt); pe = PositionEmbedding(1);
train_loader = loader(Nx*Nt, minibatch_size, sample_ratio); # rng
logger = TBLogger("content/log"); cb = cb_vp_vc; 

NNdims1 = [minibatch_size, 32, 32, 16, Nx];  NNdims2 = NNdims1;
model = NN_model(NNdims1,NNdims2);
NNparam, re = Flux.destructure(model);

# ------------------ test prediction ------------------
@time lossfunc(rand(Array(1:Nx*Nt), minibatch_size))
@time cb();

# ---------------------- train ----------------------
@time for epoch_idx in 1:10
    println("epoch $(epoch_idx):")
    @time Flux.train!(lossfunc, Flux.params(NNparam), train_loader, ADAMW(0.01), cb = cb)
end

for epoch_idx in 1:epochs-10
  println("epoch $(epoch_idx+10):")
  @time Flux.train!(lossfunc, Flux.params(NNparam), train_loader, ADAMW(0.001), cb = cb)
end
println("\ntraining done!")


# ---------------------- check performance ----------------------
#= minor change to the damping prediction as the two elements 
close to the boundaries are not examined in the PDE =#

# results(p)

# FrechetDistance
a = frdist(p[1:Nx],p0[1:Nx],xll)
b = frdist(p[Nx+2:end-1],p0[Nx+2:end-1],xll[2:end-1])

# interpolation & extrapolation
tl2 = LinRange(0,2*tmax,2*Nt); tspan2 = (0,2*tmax)
using ModelingToolkit

prob = ODEProblem(func!, x0, tspan, p);
fastprob = ODEProblem(modelingtoolkitize(prob),x0,tspan,p);
@time pred1 = solve(fastprob,ImplicitEulerExtrapolation(), p=p, saveat=tl)[1:2:Nx*2,:]
@benchmark solve(fastprob,ImplicitEulerExtrapolation(), p=p, saveat=tl)[1:2:Nx*2,:]
jetmap(pred1,"",tl)

# extrapolate
prob2 = ODEProblem(func!, x0, tspan2, p);
fastprob2 = ODEProblem(modelingtoolkitize(prob),x0,tspan2,p);
@time pred2 = solve(fastprob2,ImplicitEulerExtrapolation(), p=p, saveat=tl2)[1:2:Nx*2,:]
@benchmark solve(fastprob2,ImplicitEulerExtrapolation(), p=p, saveat=tl2)[1:2:Nx*2,:]
jetmap(pred2,"",tl)

y2 = solveODE(func!, p0, x0, solver, tspan2, tl2);
error1 = mean(abs, pred1.-y)
error2 = mean(abs, pred2.-y2)

writedlm("data/y.txt", y)
writedlm("data/y-extrapolate.txt", y2)

writedlm("data/NeuralSI-pred.txt", pred1)
writedlm("data/NeuralSI-pred2.txt", pred2)

# writedlm("data/p-nonlinear.txt", p)
# writedlm("data/p0-nonlinear.txt", p0)
