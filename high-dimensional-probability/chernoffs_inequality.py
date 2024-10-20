import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import plotly.express as px
    import altair as alt
    import pandas as pd
    import numpy as np
    import scipy
    return alt, mo, np, pd, px, scipy


@app.cell
def __(mo):
    n_input = mo.ui.number(1, 10_000)
    p = mo.ui.slider(0.0, 1.0, step=0.01)
    return n_input, p


@app.cell
def __(n_input, np, p, pd, scipy):
    ks = np.arange(n_input.value + 1)
    ps = scipy.stats.binom.pmf(k=ks, n=n_input.value, p=p.value)
    data = pd.DataFrame({"x": ks, "prob": ps})
    return data, ks, ps


@app.cell
def __(alt, data, mo):
    bar_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('x:O',  # Use ordinal scale to ensure bars touch
                axis=alt.Axis(labelAngle=0)),  # Ensure labels stay horizontal
        y='prob'
    )

    mo.ui.altair_chart(bar_chart)
    return (bar_chart,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Chernoffs Bound
        The binomal distribution represents the sum of many many random variables, each of which has a bernoulli distribution.

        The Chernoff's inequality is useful to bound the tail of a distribution (like the binomial). It tells us that the probability that a random variable deviates $X$ deviates from its mean $\mu = \text{E}[X]$ by a delta $a$ is bounded exponetially. The bound is proportionally stronger with the number of elements drawn $n$, and inversely proportional to the delta $a$.

        $$\text{Pr}[X \geq \mu + a] \leq e^{-\frac{a^2}{2n}}$$

        The Chernoff's inequality can be constructed for both sides by changing $\mu + a$ with $\mu - a$
        """
    )
    return


@app.cell
def __(mo, n_input):

    mo.md(rf"""
    For our binomial example above we want to construct a 95% confidence interval for $\mu$ around $X$. As a result $a$ will be determined from the right-hand side of the inequality.

    * n={n_input.value}
    * $e^{{-a^2 / 2n}} = 0.05 / 2$

    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        To find $a$:

        $$
        \begin{align}
        - a^2 / 2n &= \log{{0.025}} \\
        \implies -a^2 &= 2n \log 0.025 \\
        \implies a &= \sqrt{ -2 n \log 0.025}
        \end{align}
        $$
        """
    )
    return


@app.cell
def __(n_input, np):
    a = np.sqrt(-2 * n_input.value * np.log(0.025))
    a
    return (a,)


@app.cell
def __(mo, n_input):
    draw_input = mo.ui.slider(0, n_input.value)
    return (draw_input,)


@app.cell
def __(draw_input, mo):
    draw = draw_input.value
    mo.md(f"Draw = {draw_input} {draw_input.value}")
    return (draw,)


@app.cell
def __(a, draw, mo):
    mo.md(f"Chernoff's bound suggest that $\mu$ is {draw} $\pm$ {a:.3f} with 95% confidence")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Visualizing The Chernoff Confidence
        We might be interested in the probability that $\mu$ is within a deviation of size $a$ from $X$ and how that depends on the size of $a$. Let's visualize that:
        """
    )
    return


@app.cell
def __(n_input, np):
    a_range = np.linspace(0.01, n_input.value)
    rhs = np.exp(- a_range**2 / (2 * n_input.value))
    return a_range, rhs


@app.cell
def __(a_range, draw, ks, mu, ps, rhs):
    import matplotlib.pyplot as plt

    xs = [*(draw + a_range), *(draw - a_range)]
    ys = [*rhs, *rhs]
    plt.scatter(xs, ys)
    plt.vlines(mu, 0, 1, color="red", label="mu")
    plt.vlines(draw, 0, 1, color="blue", label="draw")
    plt.bar(ks, ps, label="prob[X=x]")
    plt.legend()
    plt.xlim(min(ks) - 1, max(ks) + 1)
    plt.gca()
    return plt, xs, ys


@app.cell
def __(mo, n_input, p):
    mu = n_input.value * p.value
    mo.md(f"Number of draws: {n_input}  \n probability {p} {p.value}.  \n average $\mu$ = {mu}")
    return (mu,)


if __name__ == "__main__":
    app.run()
