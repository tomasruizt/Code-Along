using Symbolics
using LinearAlgebra

include("GradientDescent.jl")

using .GradientDescent: opt

function update_x(x)
    ∇x = with_x(grad_x, x)
    return x - 0.1 * ∇x
end

using Statistics

function update(P, grads)
    ∇P = with_p(grad, P)
    avg_grad = mean([∇P, grads...])
    new_P = P - 0.001 * avg_grad
    return new_P, ∇P
end

using Plots

function plot_opt(xs)
    x1s = [x[1].val for x in xs];
    x2s = [x[2].val for x in xs];

    plot(x1s, x2s, seriestype = :scatter, xlabel = "x1", ylabel = "x2", label = "Evolution of x", alpha=0.5)
    quiver!(x1s[1:end-1], x2s[1:end-1], quiver=(diff(x1s), diff(x2s)), alpha=0.5)
    scatter!([x_star[1]], [x_star[2]], label="Target x*")

    savefig("plot_opt.png")
end

@variables x1 x2
@variables p1 p2 p3 p4;

with_x(expr, xs) = substitute(expr, Dict(x1 => xs[1], x2 => xs[2]))
with_p(x, p) = substitute(x, Dict(p1=>p[1], p2=>p[2], p3=>p[3], p4=>p[4]));

x_star = [5, 6]

A = [p1 + p2 p2 - p3; p1 + p3 p4 - p1]
b = [2 * p3, -p4]
x = A \ b

error = norm(x - x_star)
error_x = norm([x1, x2] - x_star)

grad = Symbolics.gradient(error, [p1, p2, p3, p4]);
grad_x = Symbolics.gradient(error_x, [x1, x2]);

x₁ = [1.07, -2.76]
P₁ = [1, 2, 3, 4]

errors, xs = GradientDescent.opt(P₁, update, p -> with_p(error, p), p -> with_p(x, p), n_iterations=10000);

plot_opt(xs)