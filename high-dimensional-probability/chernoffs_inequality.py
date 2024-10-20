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
    p = mo.ui.slider(0.01, 0.99, step=0.01)
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

        The Chernoff's inequality is useful to bound the tail of a distribution (like the binomial). It tells us that the probability that a random variable $X$ deviates from its mean $\mu = \mathbb{E}[X]$ by a delta $\delta$ is bounded exponetially. The Chernoff Bound comes in many forms, and a specific one exists for random variables with binomial distributions.

        The upper tail bound is: 
        $$\mathbb{P}[X \geq (1+\delta) \mu] \leq \exp(\frac{- \delta^2 \mu}{3})$$

        While the lower tail bound is:
        $$\mathbb{P}[X \leq (1-\delta) \mu] \leq \exp(\frac{- \delta^2 \mu}{2})$$

        The variable $\delta$ must be within [0, 1]. These formulas are taken from [this url](https://courses.cs.washington.edu/courses/cse312/20su/files/student_drive/6.2.pdf).
        """
    )
    return


@app.cell
def __(mo, n_input):
    draw_input = mo.ui.slider(0, n_input.value)
    return (draw_input,)


@app.cell
def __(mo):
    mo.md(
        """
        In this example we are controlling the sucessful draws from the binomial distribution. We can find out $\delta$ by the equation:

        $$\\text{draws} = (1+\delta)\mu$$

        $$\implies \\frac{\\text{draws}}{\mu} - 1 = \delta$$
        """
    )
    return


@app.cell
def __(draw_input, mo, mu):
    draw = draw_input.value
    delta = (draw / mu) - 1
    mo.md(f"""
    Draws = {draw_input} {draw_input.value}

    $\delta = {delta:.3f}$
    """)
    return delta, draw


@app.cell
def __(delta, draw, mo, mu, np):
    _prob = np.exp(- delta**2 * mu / 3)
    mo.md(f"""
    * mu = {mu:.3f}
    * $\delta$ = {delta:.3f}
    * X = {draw}

    The probability that $X \geq (1 + \delta) \mu$ = {(1 + delta) * mu:.3f} is **less** than {_prob:.3f}.

    * **Note:** Probabilities close to 1 tell almost nothing (non-informative), because all these probabilities are bounded by 1. The probability is interesting when its small, because it bounds the probability mass above the threshold (in the case of the upper tail bound).
    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Visualizing The Chernoff Confidence
        We are interested in how quickly the bound decays in the tails of the distribution. Let's visualize that.
        """
    )
    return


@app.cell
def __(mu, n_input, np):
    up_side = np.linspace(np.ceil(mu), n_input.value + 1, min(100, n_input.value)).round()
    deltas = (up_side/mu) - 1
    rhs = np.exp(- deltas**2 * mu / 3)
    return deltas, rhs, up_side


@app.cell
def __(alt, ks, mo, n_input, p, pd, rhs, scipy, up_side):
    # data.assign(name="binom")
    _df = pd.concat(
        [
            pd.DataFrame({"name": "chernoff_upper_bound", "x": up_side, "prob": rhs}),
            pd.DataFrame({"name": "binom_sf", "x": ks, "prob": scipy.stats.binom.sf(k=ks, n=n_input.value, p=p.value)})
        ], ignore_index=True
    )

    _chart = alt.Chart(_df).mark_line().encode(
        x=alt.X('x:O',  # Use ordinal scale to ensure bars touch
                axis=alt.Axis(labelAngle=0)),  # Ensure labels stay horizontal
        y='prob',
        color="name"
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def __(mo, n_input, p):
    mu = n_input.value * p.value
    mo.md(f"Number of draws: {n_input}  \n probability {p} {p.value}.  \n average $\mu$ = {mu:.3f}")
    return (mu,)


@app.cell(disabled=True, hide_code=True)
def __(ks, mu, ps, rhs, up_side):
    import matplotlib.pyplot as plt

    plt.scatter(up_side, rhs, label="chernoff upper bound")
    plt.vlines(mu, 0, 1, color="red", label="mu")
    plt.bar(ks, ps, label="prob[X=x]")
    plt.legend()
    plt.xlim(min(ks) - 1, max(ks) + 1)
    plt.gca()
    return (plt,)


if __name__ == "__main__":
    app.run()
