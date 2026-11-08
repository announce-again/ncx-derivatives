# Binomial Tree Pricing

The Cox-Ross-Rubinstein implementation provides a general-purpose numerical
pricing method for European and American options. It supports continuous
dividend yield through the risk-neutral growth term:

```text
p = (exp((r - q) dt) - d) / (u - d)
```

## Complexity

The tree uses backward induction and stores only one layer of option values.

- Runtime: `O(steps^2)`
- Memory: `O(steps)`

Within each tree layer, node spot prices are advanced by multiplication from
the lowest node instead of recomputing powers for every node. This keeps the
same asymptotic complexity while reducing the constant factor.

## Return Values

The public pricing functions return a `float` by default:

```python
price = binomial_put_price(..., american=True)
```

When `return_exercise_boundary=True`, they return `BinomialTreeResult`:

```python
result = binomial_put_price(
    ...,
    american=True,
    return_exercise_boundary=True,
)
price = result.price
boundary = result.exercise_boundary
```

European options do not have early exercise, so their exercise boundary is
reported as a tuple of `None` values. The maturity layer is also reported as
`None`; the boundary records early-exercise decisions before expiry, not the
terminal payoff.

## Numerical Limitations

- Coarse trees can show visible discretisation error. European prices should
  be compared to Black-Scholes-Merton prices using a documented tolerance.
- Raw CRR prices can oscillate between odd and even tree depths, so convergence
  is the relevant validation target rather than strict price monotonicity for
  every increase in `steps`.
- For convergence tests, compare sufficiently deep trees to the analytical
  Black-Scholes-Merton value, compare same-parity depth sequences, or compare
  adjacent odd/even averages.
- Extreme carry assumptions or very small tree depths can push the CRR
  risk-neutral probability outside `[0, 1]`. Increase `steps` in that case.
- Exercise boundaries are extracted on the discrete tree grid, so they are
  step-size approximations rather than continuous-time boundaries.
- Zero-volatility American options are still evaluated on the tree's discrete
  exercise dates. Increasing `steps` makes this discrete approximation closer
  to continuous exercise.
- The tree also uses deterministic evolution when the up and down factors round
  to the same floating-point value.
