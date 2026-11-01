# Executes the ORIGINAL CGD update routines, not a rewritten optimizer.
# Only derivative closures are analytic, avoiding historical AD dependencies.
# Their values are independently checked against JAX in the Python harness.
using LinearAlgebra, Random, DelimitedFiles
using LinearMaps, IterativeSolvers
BLAS.set_num_threads(1)
source = read("vendor/CGD/optimizers_abstract.jl", String)
source = replace(source, "using ReverseDiff" => "", "using ForwardDiff" => "")
include_string(Main, source, "vendor/CGD/optimizers_abstract.jl")
const OUT = "results/julia"
mkpath(OUT)
Random.seed!(123)
budget = 100000
etas = [0.005, 0.025, 0.1, 0.4]
open(joinpath(OUT,"environment.txt"),"w") do io
    println(io, "Julia=", VERSION)
    println(io, "LinearMaps=", pkgversion(LinearMaps))
    println(io, "IterativeSolvers=", pkgversion(IterativeSolvers))
    println(io, "Original CGD source=766fa08dc29f6e5f3170c781139fe44c42a16267")
    println(io, "RNG seed=123, full source budget=100000, epsilon=1e-6")
    println(io, "Analytic derivative closures; original optimizer bodies unchanged.")
end
for d in [20,40,60]
    U=randn(d,d); sigma=U*U'; B=rand(d,d).-0.5
    W0=B*B'; V0=U+(rand(d,d).-0.5)
    fixture=joinpath(OUT,"d$(d)"); mkpath(fixture)
    for (name,a) in [("U",U),("W0",W0),("V0",V0),("sigma",sigma)]
        writedlm(joinpath(fixture,"$(name).csv"),a,',')
    end
    gx=(w,v)->vec(reshape(v,d,d)*reshape(v,d,d)'-sigma)
    gy=(w,v)->vec(-(reshape(w,d,d)+reshape(w,d,d)')*reshape(v,d,d))
    hxy=(q,w,v)->vec(reshape(q,d,d)*reshape(v,d,d)'+reshape(v,d,d)*reshape(q,d,d)')
    hyxg=(p,w,v)->vec(-(reshape(p,d,d)+reshape(p,d,d)')*reshape(v,d,d))
    hxx=(p,w,v)->zero(p)
    hyyg=(q,w,v)->vec(-(reshape(w,d,d)+reshape(w,d,d)')*reshape(q,d,d))
    res=(w,v)->sqrt(norm(reshape(v,d,d)*reshape(v,d,d)'-sigma)^2+norm((reshape(w,d,d)+reshape(w,d,d)')/2)^2)
    for eta in etas, alg in ["CGD","OGDA","SGA","ConOpt"]
        dir=joinpath(fixture,"$(alg)_eta$(eta)");mkpath(dir)
        w=vec(copy(W0));v=vec(copy(V0));dw=zero(w);dv=zero(v)
        work=0;completed=0;status="budget";rows=Vector{Vector{Float64}}()
        push!(rows,[0.,0.,0.,res(w,v)])
        checkpoints=Set([1,10,50,100,200,500,1000])
        start=time()
        for step in 1:budget
            oldw=copy(w);oldv=copy(v)
            if alg=="CGD"
                used = iseven(step) ? CG_CGDA_x!(w,v,dw,dv,gx,hxy,gy,hyxg,eta,1e-6) : CG_CGDA_y!(w,v,dw,dv,gx,hxy,gy,hyxg,eta,1e-6)
            elseif alg=="OGDA"
                used=OGDA!(w,v,dw,dv,gx,gy,eta)
            elseif alg=="SGA"
                used=SGA!(w,v,gx,gy,hxy,hyxg,eta,1.)
            else
                used=conOpt!(w,v,gx,gy,hxx,hxy,hyyg,hyxg,eta,1.)
            end
            if work+used>budget
                w=oldw;v=oldv;break
            end
            work+=used;completed=step;r=res(w,v)
            push!(rows,[Float64(step),Float64(work),Float64(used),r])
            if alg=="CGD" && step in checkpoints
                writedlm(joinpath(dir,"state_$(step).csv"),hcat(w,v),',')
            end
            if !isfinite(r) || r>1e100
                status="diverged";break
            end
        end
        writedlm(joinpath(dir,"history.csv"),reduce(vcat,permutedims.(rows)),',')
        writedlm(joinpath(dir,"final.csv"),hcat(w,v),',')
        open(joinpath(dir,"summary.txt"),"w") do io
            println(io,"status=",status);println(io,"steps=",completed);println(io,"work=",work)
            println(io,"residual=",rows[end][4]);println(io,"seconds=",time()-start)
        end
        println("RESULT d=",d," eta=",eta," method=",alg," steps=",completed," work=",work," R=",rows[end][4]," status=",status)
        flush(stdout)
    end
end
