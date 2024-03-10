module GradientDescent

using Printf

function opt(initial_params, update_fn, err_fn, x_fn; n_iterations = 1000)
    errors = []
    xs = []
    grads = []

    Pᵢ = initial_params
    for idx in 1:n_iterations
        new_P, gradᵢ = update_fn(Pᵢ, grads)
        push!(grads, gradᵢ)
        grads = grads[max(1, end-100):end]

        if isapprox(new_P, Pᵢ, atol=1e-4)
            println("Iteration $idx. Update did not change parameters. Exiting opt()")
            return errors, xs
        end        
        Pᵢ = new_P

        if idx % 10 == 0
            errᵢ = err_fn(Pᵢ)
            xᵢ = x_fn(Pᵢ)
            push!(errors, errᵢ)
            push!(xs, xᵢ)
        end
        if idx % 100 == 0
            println("Iteration $idx. x=", [round(x.val, digits=3) for x in xᵢ])
        end
    end

    errors = [e.val for e in errors] # convert to Float64
    return errors, xs
end

end
