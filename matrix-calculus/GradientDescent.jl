module GradientDescent

using Printf

function opt(initial_params, update_fn, err_fn, x_fn; n_iterations = 1000)
    errors = []
    xs = []
    params = []

    Pᵢ = initial_params
    for idx in 1:n_iterations
        new_P, _ = update_fn(Pᵢ, [])

        if isapprox(new_P, Pᵢ, atol=1e-4)
            println("Iteration $idx. Update did not change parameters. Exiting opt()")
            break
        end       
 
        Pᵢ = new_P
        errᵢ = err_fn(Pᵢ)
        xᵢ = x_fn(Pᵢ)
        push!(errors, errᵢ)
        push!(params, Pᵢ)
        push!(xs, xᵢ)

        if isapprox(errᵢ.val, 0.0, atol=1e-4)
            println("Iteration $idx. Error is close to zero. Exiting opt()")
            break
        end

        if idx % 100 == 0
            println("Iteration $idx. x=", [round(x.val, digits=3) for x in xᵢ])
        end
    end

    errors = [e.val for e in errors] # convert to Float64
    xs = hcat([[a.val for a in x] for x in xs]...)'
    params = hcat([[a.val for a in p] for p in params]...)'
    return errors, xs, params
end

end
