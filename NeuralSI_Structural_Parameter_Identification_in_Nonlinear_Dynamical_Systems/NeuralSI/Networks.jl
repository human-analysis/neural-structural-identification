
using DifferentialEquations, DiffEqFlux, CUDA, Plots, Flux.Data, Transformers.Basic
using TensorBoardLogger, Logging

struct Split{T}
    paths::T
end
Split(paths...) = Split(paths); Flux.@functor Split; 
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths);

function loader(datalength, minibatch_size, sample_ratio, rng=123)
    index0 = randsubseq(MersenneTwister(rng), 1:datalength, sample_ratio) # random index
    train_index = cpu(index0[1:end-rem(length(index0),minibatch_size)])
    return DataLoader(train_index, batchsize=minibatch_size, shuffle=true) 
end

function def_FFNN_sigmoid(dim)
    layer = Dense(dim[1], dim[2], sigmoid)
    for i in 2:length(dim)-1
        layer = cat(layer, Dense(dim[i], dim[i+1], sigmoid), dims=1)
    end
    return Chain(layer...)
end

function def_FFNN_no_digmoid_output_vc(dim)
    layer = Dense(dim[end-1], dim[end],x->abs(x))
    for i in 2:length(dim)-1
        layer = cat(Dense(dim[end-i], dim[end-i+1], sigmoid), layer; dims=1)
    end
    return Chain(layer...)
end

function NN_model(NNdims1,NNdims2)
    NN1 = def_FFNN_sigmoid(NNdims1); 
    NN1 = Chain(NN1[:]..., x -> 0.6*x .+ 0.7f0);
    NN2 = def_FFNN_no_digmoid_output_vc(NNdims2);
    model = Chain(Split(NN1,NN2));
    return model
end

function lossfunc(index_input)
    NN_input = pe(xlmap[cpu(index_input)]')'
    vp,vc = re(NNparam)(NN_input)
    # prediction = solve(fastprob,solver,p=cat(vp,vc;dims=1),saveat=tl)[1:2:Nx*2,:];
    prediction = solveODE(func!, cat(vp,vc;dims=1), x0, solver, tspan, tl)
    global p = cat(vp,vc;dims=1)
    loss = 1e4 * mean(abs, (y .- prediction)[cpu(index_input)])
    global loss_value = loss
    return loss
end

function cb_vp_vc()
    u2 = visual_vp(p)
    u3 = visual_vc(p)
    u4 = plot(u2,u3,layout=(2,1),size=(600,500))
    display(u4)
    # prediction = solve(fastprob, solver, p=p, saveat=tl)[1:2:Nx*2,:]
    # display(jet1D(prediction))

    println("loss: ", loss_value)
    with_logger(logger) do       
        @info "loss" loss = loss_value
    end
    return false
end

function visual_vc(v;position=:topleft)
    plot(xll[2:end-1],p0[Nx+2:end-1],marker=(:circle,5), legend=position, foreground_color_legend = nothing, lw=1, label="True")
    return plot!(xll[2:end-1],v[Nx+2:end-1],label="Predict",marker=(:hex,5), lw=1, title = "Damping C")
end

function visual_vp(v;position=:bottomleft)
    plot(xll,p0[1:Nx],marker=(:circle,5), legend=position, foreground_color_legend = nothing, lw=1, label="True")
    return plot!(xll,v[1:Nx],label="Predict",marker=(:hex,5), lw=1, title = "Modulus coefficient P")
end

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
