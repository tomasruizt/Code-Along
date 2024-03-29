using Plots, ImplicitEquations

function plot_opt(xs, x_star=[5.0 6.0])
    x1s = [x[1] for x in xs];
    x2s = [x[2] for x in xs];

    plot(x1s, x2s, seriestype = :scatter, xlabel = "x1", ylabel = "x2", label = "Evolution of x", alpha=0.5)
    quiver!(x1s[1:end-1], x2s[1:end-1], quiver=(diff(x1s), diff(x2s)), alpha=0.5)
    scatter!([x_star[1]], [x_star[2]], label="Target x*")
end

function animate_opt(xs, fps = 10)
    xs_ = xs
    anim = @animate for i in 1:(size(xs_)[1])
        plot(xs_[1:i, 1], xs_[1:i, 2], xlim=(-3, 10), ylim=(-3, 10), marker=:circle, color=:blue, label="Progression", alpha=.5)
        plot!([x_star[1]], [x_star[2]], marker=:circle, color=:red, label="Target")
    end
    gif(anim, "animation.gif", fps=fps)
end

function animate_lse(ps, fps = 10)
    ps_ = ps
    anim = @animate for i in 1:(size(ps_)[1])
        p = ps_[i, :]
        plot_lse(p)
    end
    gif(anim, "lse-animation.gif", fps=fps)
end

function plot_lse(A, b, x)
    plot(line1(A, b) ⩵ 0, xlabel="x1", ylabel="x2", xlims=(-3, 10), ylims=(-3, 10))
    plot!(line2(A, b) ⩵ 0, xlims=(-3, 10), ylims=(-3, 10))
    vline!([x[1]], color=:blue)
    hline!([x[2]], color=:blue)
end

function plot_lse(p)
    A_ = A(p)
    b_ = b(p)
    x_ = x(p)
    plot_lse(A_, b_, x_)
end