from build import *

F = "07_bayesian_inference.ipynb"
cells = [
header(F, "7. Bayesian inference",
    ["Use Bayes' theorem to update probabilities in the light of data, and see why base rates matter.",
     "Understand prior, likelihood and posterior, and compute posteriors for simple problems exactly and on a grid.",
     "Interpret a credible interval, and see how it differs from a confidence interval.",
     "Understand how the prior matters when data are scarce and fades when data are plentiful.",
     "Run a simple Markov chain Monte Carlo sampler, the tool that makes Bayesian methods practical."],
    ["`stats.beta`, `stats.binomtest(...).proportion_ci`",
     "`stats.norm` for the conjugate normal model",
     "grid evaluation with `np.meshgrid`, and marginalisation by summing",
     "a hand-written Metropolis sampler using `rng.normal` and `rng.uniform`"]),

SETUP,

md(r"""
## Two views of probability

Everything so far has been **frequentist**: probability is the long-run frequency of an outcome in repeated experiments, parameters are fixed unknown constants, and probability statements are made about *data* and *procedures* (a confidence interval captures the truth in 95% of repetitions; a p-value is the probability of data as extreme as ours if $H_0$ holds). We were careful never to say "the probability that the parameter lies in this interval", because in that view it is not a random quantity.

The **Bayesian** view takes probability to be a measure of *degree of belief*, and applies it to anything uncertain, parameters and hypotheses included. Before seeing data we have a **prior** distribution for the parameters; the data update it to a **posterior** distribution, which is the complete answer to any question about the parameters. The updating rule is Bayes' theorem.

Neither view is "right". They ask different questions and, with plenty of data, usually give similar numerical answers. The Bayesian approach is more direct to interpret and handles complicated problems in a uniform way, at the cost of having to state a prior. Much modern data analysis in geophysics, geochronology and climate science is Bayesian, so you should be comfortable with both.
"""),

md(r"""
## Bayes' theorem and the base-rate trap

For two events $A$ and $B$, the definition of conditional probability gives $P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$, and hence

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)} .
$$

Here is a classic application. A seismic discrimination algorithm classifies events as earthquakes or explosions. It correctly flags 99% of explosions, and wrongly flags 2% of earthquakes. In the region of interest, one event in a thousand is an explosion. An event has just been flagged. What is the probability that it is an explosion?

$$
P(\text{expl} \mid \text{flag}) = \frac{P(\text{flag} \mid \text{expl})\,P(\text{expl})}{P(\text{flag} \mid \text{expl})\,P(\text{expl}) + P(\text{flag} \mid \text{quake})\,P(\text{quake})} .
$$
"""),

code(r'''
p_expl = 0.001
p_flag_given_expl = 0.99
p_flag_given_quake = 0.02

p_flag = p_flag_given_expl * p_expl + p_flag_given_quake * (1 - p_expl)
p_expl_given_flag = p_flag_given_expl * p_expl / p_flag
print(f"P(explosion | flagged) = {p_expl_given_flag:.3f}")
'''),

md(r"""
Only about 5% of flagged events are explosions, despite the algorithm being "99% accurate". The false alarms from the vastly more numerous earthquakes swamp the true detections. The prior probability, the **base rate**, matters enormously, and ignoring it is one of the most common errors in reasoning about evidence. Note the structure of the calculation: prior $\times$ likelihood, normalised over the alternatives.
"""),

md(r"""
## Inference for a parameter

The same rule applies to a continuous parameter $\theta$ and data $D$:

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)}, \qquad\text{or}\qquad
\underbrace{p(\theta \mid D)}_{\text{posterior}} \;\propto\; \underbrace{p(D \mid \theta)}_{\text{likelihood}} \times \underbrace{p(\theta)}_{\text{prior}} .
$$

The likelihood is exactly the function we maximised in Notebook 5. The denominator $p(D)$ does not depend on $\theta$ and is fixed by requiring the posterior to integrate to one. So Bayesian inference is: **write down the likelihood, multiply by the prior, normalise**. Everything we want (a best estimate, an interval, the probability that $\theta$ exceeds some value) is then read off the posterior.

### Example: estimating a proportion

What fraction $\theta$ of Old Faithful eruptions are "short" (under three minutes)? Each eruption is short with probability $\theta$, so $k$ short eruptions out of $n$ is a binomial observation with likelihood $\propto \theta^k (1-\theta)^{n-k}$. A convenient prior is the **beta distribution**, $p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$, because multiplying it by the binomial likelihood gives another beta distribution:

$$
\text{prior } \mathrm{Beta}(\alpha, \beta) \;\xrightarrow{\;k \text{ of } n\;}\; \text{posterior } \mathrm{Beta}(\alpha + k,\; \beta + n - k).
$$

A prior that leads to a posterior of the same family is called **conjugate**, and makes the calculation trivial. $\mathrm{Beta}(1, 1)$ is the uniform distribution on $[0, 1]$, a natural choice when we claim no prior knowledge. Let us watch the posterior sharpen as eruptions are observed one after another.
"""),

code(r'''
faithful = pd.read_csv(DATA + "old_faithful.csv")
short = (faithful["eruptions"] < 3).to_numpy()

alpha0, beta0 = 1, 1                  # uniform prior
theta = np.linspace(0, 1, 500)
fig, ax = plt.subplots()
for n in [0, 5, 20, 100, len(short)]:
    k = short[:n].sum()
    post = stats.beta(alpha0 + k, beta0 + n - k)
    ax.plot(theta, post.pdf(theta), label=f"after {n} eruptions ({k} short)")
ax.set(xlabel="θ, fraction of short eruptions", ylabel="posterior density"); ax.legend()
plt.show()

n, k = len(short), short.sum()
post = stats.beta(alpha0 + k, beta0 + n - k)
lo, hi = post.interval(0.95)
print(f"posterior mean = {post.mean():.3f},  95% credible interval = [{lo:.3f}, {hi:.3f}]")
print(f"P(θ > 0.4) = {post.sf(0.4):.3f}")
ci = stats.binomtest(int(k), n).proportion_ci(0.95, method="wilson")
print(f"for comparison, frequentist 95% confidence interval = [{ci.low:.3f}, {ci.high:.3f}]")
'''),

md(r"""
The interval $[0.30, 0.42]$ is a **credible interval**: there is a 95% probability, given the data and the prior, that $\theta$ lies in it. That is the statement everyone *wants* a confidence interval to make, and in the Bayesian framework it is legitimate. We can equally ask for the probability that $\theta$ exceeds 0.4, a question with no frequentist counterpart. The frequentist confidence interval is numerically almost identical here; with this much data the two approaches agree, and only their interpretation differs.
"""),

md(r"""
### Example: the mean of normal measurements

Suppose we make $n$ measurements $x_i$ of a quantity $\mu$ with known error $\sigma$, and our prior for $\mu$ is normal with mean $m_0$ and standard deviation $s_0$. The normal prior is conjugate to the normal likelihood and the posterior is normal with

$$
\frac{1}{s_n^2} = \frac{1}{s_0^2} + \frac{n}{\sigma^2}, \qquad
m_n = s_n^2 \left( \frac{m_0}{s_0^2} + \frac{n\bar{x}}{\sigma^2} \right).
$$

Compare this with the weighted mean of Notebook 2: the posterior mean is the inverse-variance weighted average of the prior mean and the data mean, and the precisions (inverse variances) add. **The prior behaves exactly like one extra measurement** of $\mu$ with uncertainty $s_0$. This makes the influence of the prior easy to judge: it matters when its precision is comparable to that of the data, and becomes irrelevant as $n$ grows.

As an example: the density of a granite is to be measured. From the literature we take a prior of $2.70 \pm 0.05$ g cm$^{-3}$. Our balance gives measurements with $\sigma = 0.03$ g cm$^{-3}$.
"""),

code(r'''
m0, s0 = 2.70, 0.05
sigma = 0.03
true_density = 2.62
x = stats.norm(true_density, sigma).rvs(20, random_state=rng)

def posterior(x):
    n = len(x)
    prec = 1/s0**2 + n/sigma**2
    return stats.norm(loc=(m0/s0**2 + n*x.mean()/sigma**2) / prec, scale=np.sqrt(1/prec))

mu = np.linspace(2.5, 2.85, 400)
fig, ax = plt.subplots()
ax.plot(mu, stats.norm(m0, s0).pdf(mu), "k--", label="prior")
for n in [1, 3, 20]:
    ax.plot(mu, posterior(x[:n]).pdf(mu), label=f"posterior after n = {n}")
ax.axvline(true_density, color="0.5", lw=1)
ax.set(xlabel="density (g/cm³)", ylabel="density of probability"); ax.legend()
plt.show()

for n in [1, 3, 20]:
    P = posterior(x[:n])
    print(f"n = {n:2d}: data mean = {x[:n].mean():.3f} ± {sigma/np.sqrt(n):.3f};  posterior = {P.mean():.3f} ± {P.std():.3f}")
'''),

md(r"""
With a single measurement the posterior sits between the prior and the datum, weighted towards the datum because it is the more precise. By $n = 20$ the prior has almost no influence and the posterior mean is essentially the sample mean. This is the general pattern: **a prior matters when data are scarce, and is swamped when they are plentiful**. If a result depends strongly on the prior, that is the analysis telling you that the data alone do not settle the question, which is worth knowing.
"""),

md(r"""
## Posteriors on a grid: the straight line revisited

Conjugate priors are rare luxuries. For a general problem with a few parameters, the simplest approach is to evaluate the posterior numerically on a grid. Take the straight-line fit of Notebook 5, with normal errors and flat priors on $a$ and $b$. Then

$$
p(a, b \mid D) \propto L(a, b) \propto \exp\!\left[-\tfrac{1}{2}\chi^2(a, b)\right],
$$

so the posterior is a simple function of the same $\chi^2$ surface we contoured before. From the two-dimensional posterior we obtain the **marginal** posterior of the slope by integrating out the intercept, $p(a \mid D) = \int p(a, b \mid D)\, \mathrm{d}b$, which on a grid is a sum over columns. Marginalisation is the Bayesian way of handling **nuisance parameters**, those we must include but do not care about; it averages over their uncertainty rather than fixing them at their best values.
"""),

code(r'''
n = 50
x = np.linspace(0, 1, n)
sigma = 0.2
y = 2*x - 4 + stats.norm(0, sigma).rvs(n, random_state=rng)

aa = np.linspace(1.4, 2.6, 300)
bb = np.linspace(-4.35, -3.65, 300)
A, B = np.meshgrid(aa, bb)
chi2 = np.sum(((y[:, None, None] - (A[None]*x[:, None, None] + B[None])) / sigma)**2, axis=0)
post = np.exp(-0.5 * (chi2 - chi2.min()))
post /= post.sum() * (aa[1]-aa[0]) * (bb[1]-bb[0])          # normalise to unit integral

marg_a = post.sum(axis=0) * (bb[1]-bb[0])                    # integrate over b
marg_b = post.sum(axis=1) * (aa[1]-aa[0])                    # integrate over a

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].contourf(A, B, post, levels=20, cmap="Blues")
ax[0].set(xlabel="slope a", ylabel="intercept b", title="joint posterior p(a, b | data)")
ax[1].plot(aa, marg_a)
ax[1].set(xlabel="slope a", ylabel="p(a | data)", title="marginal posterior of the slope")
plt.show()

cdf_a = np.cumsum(marg_a) * (aa[1]-aa[0])
lo, hi = np.interp([0.025, 0.975], cdf_a, aa)
a_mean = np.sum(aa * marg_a) * (aa[1]-aa[0])
a_sd = np.sqrt(np.sum((aa - a_mean)**2 * marg_a) * (aa[1]-aa[0]))
print(f"posterior for the slope: mean = {a_mean:.3f}, sd = {a_sd:.3f}, 95% credible interval = [{lo:.3f}, {hi:.3f}]")
lr = stats.linregress(x, y)
print(f"least squares:           a = {lr.slope:.3f} ± {np.sqrt(sigma**2 / np.sum((x - x.mean())**2)):.3f} (standard error with known σ)")
'''),

md(r"""
With flat priors the posterior mean coincides with the least-squares estimate, the posterior standard deviation with the standard error, and the 95% credible interval with the 95% confidence interval. For linear models with normal errors and flat priors, Bayesian and frequentist answers are numerically identical; the difference lies entirely in what the interval *means*.

### An informative prior

Now suppose physics tells us, independently of these data, that the slope should be $2.3 \pm 0.1$. We multiply the likelihood by this prior.
"""),

code(r'''
prior_a = stats.norm(2.3, 0.1)
post2 = post * prior_a.pdf(A)
post2 /= post2.sum() * (aa[1]-aa[0]) * (bb[1]-bb[0])
marg_a2 = post2.sum(axis=0) * (bb[1]-bb[0])

fig, ax = plt.subplots()
ax.plot(aa, prior_a.pdf(aa), "k--", label="prior on a")
ax.plot(aa, marg_a, label="posterior, flat prior")
ax.plot(aa, marg_a2, label="posterior, informative prior")
ax.set(xlabel="slope a", ylabel="density"); ax.legend()
plt.show()

a_mean2 = np.sum(aa * marg_a2) * (aa[1]-aa[0])
a_sd2 = np.sqrt(np.sum((aa - a_mean2)**2 * marg_a2) * (aa[1]-aa[0]))
print(f"flat prior:        a = {a_mean:.3f} ± {a_sd:.3f}")
print(f"informative prior: a = {a_mean2:.3f} ± {a_sd2:.3f}")
'''),

md(r"""
The posterior moves towards the prior and narrows, since the prior adds information. Whether this is legitimate depends entirely on whether the prior is *justified*: if the "physics" was wrong, the prior has pulled the answer away from the truth. Priors should be stated explicitly and their influence tested by varying them (a **sensitivity analysis**). A flat prior is not automatically "objective" either: flat in $a$ is not flat in $1/a$ or $\ln a$, so even the uninformative choice involves a decision about parameterisation. With good data these subtleties do not matter; with poor data they do, and the Bayesian framework at least makes them visible.
"""),

md(r"""
## Markov chain Monte Carlo

A grid is fine for two parameters. For ten it is hopeless: a grid of 100 points per dimension has $10^{20}$ cells. Real problems, with dozens or thousands of parameters, need a different approach. **Markov chain Monte Carlo (MCMC)** generates a sequence of random parameter values whose distribution *is* the posterior, without ever normalising it. From the samples, posterior means, intervals and marginals are just averages and histograms, exactly as with Monte Carlo error propagation in Notebook 3.

The simplest MCMC method is the **Metropolis algorithm**. Starting from some $\boldsymbol{\theta}$:

1. Propose a move to $\boldsymbol{\theta}'$ by adding a random step.
2. Compute the ratio of posterior densities, $r = p(\boldsymbol{\theta}' \mid D) / p(\boldsymbol{\theta} \mid D)$. Only the *unnormalised* posterior is needed.
3. If $r \ge 1$ accept the move. Otherwise accept it with probability $r$; if rejected, stay put (and record the current point again).
4. Repeat.

The chain wanders preferentially into regions of high posterior density and, after an initial **burn-in** period, its visits are distributed according to the posterior. Here it is for the straight-line problem, in a dozen lines. In practice we work with the logarithm of the posterior to avoid overflow, so the acceptance test compares $\ln r$ with the logarithm of a uniform random number.
"""),

code(r'''
def log_posterior(theta):
    a, b = theta
    return -0.5 * np.sum(((y - (a*x + b)) / sigma)**2)          # flat priors: log posterior = -χ²/2

n_steps = 20_000
step = np.array([0.08, 0.05])                                 # proposal standard deviations
theta = np.array([1.0, -3.0])                                 # a deliberately poor starting point
lp = log_posterior(theta)
chain = np.empty((n_steps, 2)); accepted = 0
for i in range(n_steps):
    proposal = theta + step * rng.normal(size=2)
    lp_prop = log_posterior(proposal)
    if np.log(rng.uniform()) < lp_prop - lp:
        theta, lp = proposal, lp_prop
        accepted += 1
    chain[i] = theta
print(f"acceptance rate = {accepted / n_steps:.2f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(chain[:2000, 0], lw=0.5); ax[0].set(xlabel="step", ylabel="slope a", title="trace: first 2000 steps")
burn = 1000
ax[1].plot(chain[burn:, 0], chain[burn:, 1], ".", ms=1, alpha=0.2); ax[1].set(xlabel="a", ylabel="b", title="samples after burn-in")
ax[2].hist(chain[burn:, 0], bins=50, density=True, label="MCMC samples")
ax[2].plot(aa, marg_a, "C1", label="grid marginal"); ax[2].set(xlabel="slope a", title="marginal of a"); ax[2].legend()
plt.tight_layout(); plt.show()

s = chain[burn:]
print(f"MCMC: a = {s[:, 0].mean():.3f} ± {s[:, 0].std():.3f},  b = {s[:, 1].mean():.3f} ± {s[:, 1].std():.3f},"
      f"  95% interval for a = [{np.percentile(s[:, 0], 2.5):.3f}, {np.percentile(s[:, 0], 97.5):.3f}]")
'''),

md(r"""
The trace shows the chain moving from its poor starting point into the high-probability region within a few hundred steps, after which it fluctuates around the posterior. Discarding this burn-in, the histogram of the samples reproduces the grid marginal, and the marginal of $b$ comes free.

Some practical points. The step size matters: too small and the chain crawls, too large and almost every proposal is rejected; an acceptance rate of 20–50% is a good sign. Successive samples are correlated, so 19,000 samples are worth fewer independent draws; run the chain longer than you think necessary and check that different starting points give the same answer. And the Metropolis algorithm, while adequate here, is far from the state of the art: packages such as `emcee`, `PyMC` and `Stan` implement much more efficient samplers and are what you would use in practice. The principle is the same.
"""),

md(r"""
## Bayesian and frequentist: a comparison

| | Frequentist | Bayesian |
|---|---|---|
| Parameters are | fixed unknown constants | uncertain quantities with distributions |
| Probability applies to | data and procedures | anything uncertain, including parameters |
| Prior information | not used formally | required, stated as a prior distribution |
| Point estimate | maximum likelihood | posterior mean, median or mode |
| Interval | confidence: covers the truth in 95% of repetitions | credible: 95% probability the truth is inside |
| Nuisance parameters | profile (optimise over them) | marginalise (integrate over them) |
| Model checking | hypothesis tests, p-values | posterior predictive checks, Bayes factors |
| Main computation | optimisation | integration, usually by MCMC |

With plentiful data and weak priors the two give the same numbers, and most working scientists move between them as convenient. The Bayesian approach is more natural when prior information is genuinely available, when parameters are many and the question is about a few, and when the answer needed is a probability. The frequentist approach needs no prior, and its tests and error rates are well understood. Whichever you use, say which, and say what your intervals mean.

**Further reading.** D. S. Sivia, *Data Analysis: A Bayesian Tutorial* (Oxford) is short and written for physical scientists.
"""),

md(r"""
## Exercises
"""),

exercise(1, r"""
Repeat the beta-binomial calculation for the fraction of short eruptions with a strongly informative prior $\mathrm{Beta}(20, 20)$ (which believes the fraction is close to 0.5). Compare the posteriors from the flat prior and this prior after 10 eruptions and after all 272. Plot them. When does the prior matter?
"""),
scratch(),
solution(r"""
```python
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, n in zip(ax, [10, len(short)]):
    k = short[:n].sum()
    for (a0, b0), name in [((1, 1), "flat prior"), ((20, 20), "Beta(20, 20) prior")]:
        P = stats.beta(a0 + k, b0 + n - k)
        a.plot(theta, P.pdf(theta), label=f"{name}: mean {P.mean():.3f}")
    a.set(xlabel="θ", title=f"after {n} eruptions ({k} short)"); a.legend()
plt.show()
```

After 10 eruptions the two posteriors differ substantially: the informative prior is worth 40 pseudo-observations and dominates. After 272 the data overwhelm it and the posteriors nearly coincide.
"""),

exercise(2, r"""
*A conjugate model for the Gutenberg–Richter $b$-value.* In Notebook 5 the magnitudes above $M_c$ were modelled as exponential with rate $\beta = b \ln 10$, i.e. likelihood $\prod_i \beta e^{-\beta m_i}$ where $m_i = M_i - M_c$. The gamma distribution is conjugate: a prior $\mathrm{Gamma}(\alpha_0, \text{rate } \lambda_0)$ gives the posterior $\mathrm{Gamma}(\alpha_0 + n, \text{rate } \lambda_0 + \sum_i m_i)$. Using the whole catalogue with $M_c = 4.95$ and a weak prior ($\alpha_0 = 1$, $\lambda_0 = 0.01$), find the posterior mean and 95% credible interval for $b$, and compare with the maximum likelihood value and its standard error $b/\sqrt{n}$. (`stats.gamma(a, scale=1/rate)`.)
"""),
scratch(),
solution(r"""
```python
quakes = pd.read_csv(DATA + "usgs_earthquakes_M5_2014_2023.csv")
m = quakes["mag"].to_numpy() - 4.95
n = len(m)
post_beta = stats.gamma(1 + n, scale=1/(0.01 + m.sum()))       # posterior for β = b ln 10
b_samples = post_beta.rvs(100_000, random_state=rng) / np.log(10)
lo, hi = np.percentile(b_samples, [2.5, 97.5])
b_ml = np.log10(np.e) / m.mean()
print(f"posterior for b: mean = {b_samples.mean():.4f}, sd = {b_samples.std():.4f}, 95% credible interval = [{lo:.4f}, {hi:.4f}]")
print(f"maximum likelihood: b = {b_ml:.4f} ± {b_ml/np.sqrt(n):.4f}")
```

With 17,000 events the posterior is very sharp, and its mean and standard deviation agree with the maximum likelihood estimate and its standard error to three decimal places. (Sampling from the posterior for $\beta$ and transforming to $b$ is the easiest way to get the interval for $b$; one could also use `post_beta.interval(0.95)` and divide by $\ln 10$.)
"""),

exercise(3, r"""
Modify the Metropolis sampler to sample the posterior for the $b$-value of the previous exercise, using the log-likelihood $n \ln\beta - \beta \sum_i m_i$ and a flat prior on $\beta > 0$ (return `-np.inf` for $\beta \le 0$). Check that the histogram of the samples matches the gamma posterior. Try a step size that is far too large and one far too small and look at the trace plots.
"""),
scratch(),
solution(r"""
```python
S = m.sum()
def log_post_beta(beta):
    return n*np.log(beta) - beta*S if beta > 0 else -np.inf

def metropolis(step, n_steps=20_000, start=1.0):
    beta, lp = start, log_post_beta(start)
    chain = np.empty(n_steps); acc = 0
    for i in range(n_steps):
        prop = beta + step*rng.normal()
        lpp = log_post_beta(prop)
        if np.log(rng.uniform()) < lpp - lp:
            beta, lp, acc = prop, lpp, acc + 1
        chain[i] = beta
    return chain, acc/n_steps

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
for a, step in zip(ax, [0.0005, 0.03, 2.0]):
    chain, rate = metropolis(step)
    a.plot(chain[:3000], lw=0.5); a.set(title=f"step {step}: acceptance {rate:.2f}", xlabel="step", ylabel="β")
plt.tight_layout(); plt.show()

chain, _ = metropolis(0.03)
bs = chain[2000:] / np.log(10)
fig, ax = plt.subplots()
ax.hist(bs, bins=50, density=True, label="MCMC")
bb_ = np.linspace(bs.min(), bs.max(), 200)
ax.plot(bb_, post_beta.pdf(bb_*np.log(10))*np.log(10), "C1", label="gamma posterior")
ax.set(xlabel="b"); ax.legend(); plt.show()
print(f"MCMC: b = {bs.mean():.4f} ± {bs.std():.4f}")
```

The tiny step gives a chain that accepts almost everything but moves so slowly that 3000 steps do not even reach the posterior; the huge step is rejected almost always and the trace is a staircase. The intermediate step mixes well and its histogram matches the exact gamma posterior.
"""),
]

build(F, cells)
