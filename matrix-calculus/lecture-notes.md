## Revisit: Taylor Expansion
Of a function $f(x)$ around a point $x_0$ is given by:
$$
f(x) = f(x_0) + f'(x_0) (x - x_0) + \frac{f''(x_0)}{2!} (x - x_0)^2 + \ldots
$$
Its the infinite sum of terms of the form:
$$
f(x) = \sum_{i=0}^{n} \frac{f^{(n)}(x_0)}{n!} (x - x_0)^n
$$

This leads to the linearization formula:

## Revistit: Linearization Formula
We drop all terms of order greater than 1:

$$
\begin{align*}
f(x) - f(x_0) &= f'(x_0) (x - x_0) + o(||x - x_0||) \\
&\approx f'(x_0) (x - x_0)    
\end{align*}
$$
Solving for $f(x)$:
$$
f(x) \approx f(x_0) + f'(x_0) (x - x_0)
$$

## Revisit: Differential Formula

$$
\begin{align*}
f'(x) &= \frac{f(x + dx) - f(x)}{dx} = \frac{df}{dx} \\ \\
\implies df &= f(x + dx) - f(x) \\
\implies df &= f'(x) dx \\ \\
\implies f(x + dx) &= f(x) + f'(x) dx
\end{align*}
$$

# Integrals as Linear Operators
An operator $A$ is linear if it satisfies the following properties:
1. $A(\alpha x) = \alpha A(x)$
2. $A(x + y) = A(x) + A(y)$

For the integral $A(f) = \int f(x) dx$:
$$
\begin{align*}
A(\alpha f) 
&= \int \alpha f(x) dx \\
&= \alpha \int f(x) dx \qquad \text{(pull out $\alpha$)}\\
&= \alpha A(f)
\end{align*}
$$
and

$$
\begin{align*}
A(f + g) 
&= \int f(x) + g(x) dx \\
&= \int f(x) dx + \int g(x) dx \\
&= A(f) + A(g)
\end{align*}
$$

### More Complicated Integral I
Let $A(f) = \int cos(x) f(x) dx$. Is $A(f)$ a linear operator?

First rule:
$$
\begin{align*}
A(\alpha f)
&= \int cos(x) \alpha f(x) dx \\
&= \alpha \int cos(x) f(x) dx \qquad \text{(pull out)}\\
&= \alpha A(f)
\end{align*}
$$

Second rule:
$$
\begin{align*}
A(f + g)
&= \int cos(x) (f(x) + g(x)) dx \\
&= \int cos(x) f(x) + cos(x) g(x) dx \\
&= \int cos(x) f(x) dx + \int cos(x) g(x) dx \qquad \text{(separate integral)}\\
&= A(f) + A(g)
\end{align*}
$$

### More Complicated Integral II
Let $A(f) = \int f(x)^2 dx$. Is $A(f)$ a linear operator?
First rule:
$$
\begin{align*}
A(\alpha f) 
&= \int (\alpha f(x))^2 dx \\
&= \alpha^2 \int f(x)^2 dx \\
&= \alpha^2 A(f) \\
&\neq \alpha A(f)
\end{align*}
$$
**Answer:** $A(f)$ is not a linear operator.

**Note**: Not every integral is a linear operator.

# Adjoint Method
$$
\begin{align*}
A(p) x(p) &= b(p) \\
         \frac{d}{dp} A x &= \frac{d}{dp} b \\
\frac{dA}{dp} x + A \frac{dx}{dp} &= \frac{db}{dp} \\
    A_p x + A x_p &= b_p \\ \\

\implies          A x_p &= b_p - A_p x \\
\implies            x_p &= A^{-1} (b_p - A_p x)
\end{align*}
$$

# Examples
## Example 1
Given $f(x) = x^Tx$, compute the the differential $df$, at $x_0$ = [3, 4].

$$
\begin{align*}
df & = d(x^T)x + x^Td(x) \\
& = x^Tdx + x^Tdx \\
& = 2x^Tdx \\

\implies df(x_0) & = 2x_0^Tdx \\ \\

\implies df & = 2 <x, dx> \\
\implies \nabla f &= 2x  \\
\end{align*}
$$

## Example 8
Let $f(x) = x^TAx$. Compute the differential $df$, the derivative $f'(x)$, and the gradient $\nabla f$.

$$
\begin{align*}
df
& = d(x^TAx) \\
& = d(x^T)Ax + x^T d(Ax) \\
& = dx^TAx + x^T \underbrace{dA}_0 x + x^TAdx \\
& = x^TA^Tdx + 0 + x^TAdx \\
& = x^T(A^T + A)dx \\ \\
\implies \nabla f &= (A^T + A)x \\
\implies f'(x) 
&= (\nabla f)^T \\
&= x^T(A^T + A) \\
\end{align*}
$$

