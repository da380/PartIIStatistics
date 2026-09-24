from build import *

F = "09_inverse_problems.ipynb"
cells = [
header(F, "9. Inverse problems (optional)",
    ["Understand the difference between fitting a few parameters and inferring a function from finite data.",
     "See why an inverse problem can have no unique answer even with error-free data, and what the null space is.",
     "Know what minimum-norm and damped (regularised) solutions are, and that each is a *choice*.",
     "Understand resolution: an estimate is a blurred version of the truth, and the blurring can be computed.",
     "See how the Bayesian view turns regularisation into a prior, and gives honest uncertainties through posterior samples."],
    ["`np.linalg.svd`, `np.linalg.matrix_rank`, `np.linalg.pinv`, `np.linalg.solve`",
     "`np.random.multivariate_normal` (through `rng.multivariate_normal`) for posterior samples",
     "the `pygeoinf` library (optional final section) for the same ideas in function space"]),

SETUP,

md(r"""
*This notebook is optional and a little more advanced than the others. It assumes Notebooks 5 and 7.*

## Forward and inverse problems

Much of geophysics consists of inferring something you cannot see from measurements made at a distance: the Earth's interior from seismic travel times, density from gravity, a thermal history from isotope ratios, past climate from proxies. In each case the physics gives a **forward problem**: given the model, predict the data. The **inverse problem** is the reverse, and it is where the difficulty lies.

We have already solved inverse problems of a kind. Fitting a straight line (Notebook 5) infers two parameters from fifty data. The situation there was *overdetermined*: more data than unknowns, so the difficulty was to reconcile inconsistent data, and the answer was unique. In most geophysical problems the unknown is a *function*, seismic velocity at every point in the Earth, say, which has infinitely many degrees of freedom, while the data are finite in number. Such problems are **underdetermined**: infinitely many models predict exactly the same data. The central message of this notebook is that

> **an underdetermined inverse problem has no unique solution even when the data are exact.** What we call "the solution" is always a choice, and the choice should be made consciously and stated.

Everything below is for **linear** problems, where the data depend linearly on the model: $\mathbf{d} = \mathbf{G}\mathbf{m} + \mathbf{e}$, with $\mathbf{m}$ the vector of model parameters, $\mathbf{G}$ a known matrix from the physics, and $\mathbf{e}$ the data errors. Nonlinear problems are usually attacked by repeated linearisation, so the ideas carry over.
"""),

md(r"""
## A toy tomography problem

A square block of rock is divided into $N \times N$ cells, each with an unknown **slowness** (the reciprocal of seismic velocity). Straight rays cross the block from sources to receivers, and we measure the travel time of each ray, which is the sum over the cells it crosses of the path length in the cell times the slowness there:

$$
t_k = \sum_{j} G_{kj}\, m_j, \qquad G_{kj} = \text{length of ray } k \text{ in cell } j .
$$

This is a miniature version of seismic tomography. The function below builds $\mathbf{G}$ for any set of straight rays by sampling each ray finely and accumulating the length in each cell. We start with a very sparse experiment: one horizontal and one vertical ray through each row and column of cells.
"""),

code(r'''
N = 12                                   # the block is N x N cells on the unit square
M = N * N                                # number of model parameters (cell slownesses)

def ray_matrix(rays, N, n_pts=4000):
    """Rows: rays, columns: cells (index i + N*j for column i, row j). Entries: path length in cell."""
    G = np.zeros((len(rays), N * N))
    for k, ((x0, y0), (x1, y1)) in enumerate(rays):
        t = (np.arange(n_pts) + 0.5) / n_pts
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        i = np.minimum((x * N).astype(int), N - 1)
        j = np.minimum((y * N).astype(int), N - 1)
        np.add.at(G, (k, i + N * j), np.hypot(x1 - x0, y1 - y0) / n_pts)
    return G

def show(m, ax, title="", cmap="RdBu_r", **kw):
    """Plot a model vector as an image of the block."""
    im = ax.imshow(m.reshape(N, N), origin="lower", extent=(0, 1, 0, 1), cmap=cmap, **kw)
    ax.set(title=title, xticks=[], yticks=[]); return im

c = (np.arange(N) + 0.5) / N
rays = [((0, yc), (1, yc)) for yc in c] + [((xc, 0), (xc, 1)) for xc in c]     # N horizontal + N vertical rays
G = ray_matrix(rays, N)
print(f"{len(rays)} rays, {M} unknowns;  G has shape {G.shape}")

# A "true" slowness model: uniform background with a slow blob and a fast blob
X, Y = np.meshgrid(c, c)
m_true = (1.0 + 0.3 * np.exp(-((X - 0.3)**2 + (Y - 0.65)**2) / 0.02)
              - 0.3 * np.exp(-((X - 0.7)**2 + (Y - 0.3)**2) / 0.02)).ravel()
d_exact = G @ m_true                                                          # error-free travel times

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
im = show(m_true, ax[0], "true slowness", vmin=0.7, vmax=1.3); plt.colorbar(im, ax=ax[0])
for (x0, y0), (x1, y1) in rays:
    ax[1].plot([x0, x1], [y0, y1], "C0", lw=1)
ax[1].set(title="ray paths", xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
plt.show()
'''),

md(r"""
## Non-uniqueness and the null space

With 24 travel times and 144 unknowns we clearly cannot determine every cell. But the situation is worse than "not enough equations": the equations we have are not even independent. The sum of the horizontal travel times equals the sum of the vertical ones (both are the total slowness times the cell size), so only 23 of the 24 data carry independent information.

The **singular value decomposition** (SVD), $\mathbf{G} = \mathbf{U}\mathbf{S}\mathbf{V}^T$, is the tool for seeing this. The columns of $\mathbf{V}$ are orthonormal directions in model space; each is mapped by $\mathbf{G}$ onto the corresponding column of $\mathbf{U}$, scaled by its singular value $s_i$. The number of non-zero singular values is the **rank** of $\mathbf{G}$, the number of independent pieces of information in the data. The remaining directions in model space form the **null space**: models that produce *no data at all*. Adding any null-space model to a solution leaves the predicted data unchanged.
"""),

code(r'''
U, s, Vt = np.linalg.svd(G)
rank = np.linalg.matrix_rank(G)
print(f"rank of G = {rank};  null space has dimension {M - rank}")
print("singular values:", np.round(s, 3))

null_space = Vt[rank:]                     # rows are orthonormal null-space models
z = null_space[3]                          # any one of them (or any combination)
print(f"\nlargest |travel-time| produced by a null-space model: {np.abs(G @ z).max():.1e}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
show(m_true, ax[0], "true model", vmin=0.7, vmax=1.3)
show(z, ax[1], "a null-space model: produces zero data")
show(m_true + 1.5 * z, ax[2], "true model + 1.5 × null-space model", vmin=0.7, vmax=1.3)
plt.show()
print(f"maximum difference in predicted travel times between left and right models: {np.abs(G @ m_true - G @ (m_true + 1.5*z)).max():.1e}")
'''),

md(r"""
The models on the left and right predict *identical* travel times, to machine precision, yet one is smooth and the other is riddled with structure. No amount of measurement precision can distinguish them: the difference between them is invisible to this experiment. A 121-dimensional family of models fits the data exactly, and the data cannot tell them apart.

This is the sense in which the inverse problem "has no answer". It is not a matter of noise or of insufficient care. It follows from the geometry of the experiment, and the only cures are more (or better placed) data or extra information from outside the data.
"""),

md(r"""
## Picking a solution: the minimum-norm estimate

If we must choose one model from the family that fits the data, we need a criterion. The most common default is the **minimum-norm** solution: of all models that fit the data, the one with the smallest $\|\mathbf{m}\|$ (or, after subtracting a reference model, the one closest to it). It contains no null-space component, since adding one could only increase the norm. It is computed with the **pseudo-inverse** $\mathbf{G}^{+} = \mathbf{V}\mathbf{S}^{-1}\mathbf{U}^T$, inverting only the non-zero singular values.
"""),

code(r'''
m0 = np.ones(M)                                        # reference model: uniform slowness 1
m_mn = m0 + np.linalg.pinv(G) @ (d_exact - G @ m0)     # minimum-norm perturbation about the reference

print(f"data misfit of minimum-norm model: {np.abs(G @ m_mn - d_exact).max():.1e}")
print(f"norm of perturbation: minimum-norm {np.linalg.norm(m_mn - m0):.3f},  true model {np.linalg.norm(m_true - m0):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
show(m_true, ax[0], "true model", vmin=0.7, vmax=1.3)
show(m_mn, ax[1], "minimum-norm model (exact data)", vmin=0.7, vmax=1.3)
plt.show()
'''),

md(r"""
The minimum-norm model fits the data perfectly and looks nothing like the truth: with only row and column sums available, it can only paint stripes. It is the *simplest* model consistent with the data in a specific sense, and simplicity is a reasonable preference, but nothing in the data says the Earth is simple. The estimate reflects the experiment's geometry as much as the target.
"""),

md(r"""
## A better experiment, noise, and instability

Let us add a fan of crossing rays, from each of $N$ points on the left edge to each of $N$ points on the right edge. That gives $N^2 + 2N = 168$ rays for 144 unknowns, so the problem is now formally overdetermined. Is it solved?
"""),

code(r'''
edge = (np.arange(N) + 0.5) / N
rays2 = rays + [((0, a), (1, b)) for a in edge for b in edge]
G2 = ray_matrix(rays2, N)
U2, s2, Vt2 = np.linalg.svd(G2)
print(f"{len(rays2)} rays;  rank of G2 = {np.linalg.matrix_rank(G2)}")

fig, ax = plt.subplots()
ax.semilogy(np.arange(1, len(s) + 1), s, "o", label="24 rays")
ax.semilogy(np.arange(1, len(s2) + 1), s2, "s", ms=3, label="168 rays")
ax.set(xlabel="index", ylabel="singular value"); ax.legend()
plt.show()
'''),

md(r"""
Two things have happened. The rank has risen to 131, but not to 144: even 168 rays leave a 13-dimensional null space, because all the fan rays run broadly left to right and certain patterns of vertical variation are invisible to every one of them. And among the directions that *are* constrained, the singular values now span more than two orders of magnitude. The small ones correspond to model patterns that affect the data only weakly. In the inverse, those patterns are multiplied by $1/s_i$, so small errors in the data become large errors in the model. A problem with a wide range of singular values is **ill-conditioned**; for practical purposes it is still underdetermined, because real data always have errors.

To see this, add noise to the travel times and build the solution one singular vector at a time, from the best-constrained direction to the worst. Keeping the first $k$ terms is called **truncated SVD**:

$$
\hat{\mathbf{m}}_k = \mathbf{m}_0 + \sum_{i=1}^{k} \frac{\mathbf{u}_i^T(\mathbf{d} - \mathbf{G}\mathbf{m}_0)}{s_i}\,\mathbf{v}_i .
$$
"""),

code(r'''
sigma_d = 0.005                                                   # travel-time error, about 0.5% of a typical time
d_noisy = G2 @ m_true + stats.norm(0, sigma_d).rvs(len(rays2), random_state=rng)
rank2 = np.linalg.matrix_rank(G2)

coef = (U2.T @ (d_noisy - G2 @ m0))[:rank2] / s2[:rank2]         # coefficient of each singular vector
def tsvd(k):
    return m0 + Vt2[:k].T @ coef[:k]

ks = np.arange(1, rank2 + 1)
model_error = [np.linalg.norm(tsvd(k) - m_true) for k in ks]
data_misfit = [np.sum(((d_noisy - G2 @ tsvd(k)) / sigma_d)**2) for k in ks]

fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
for a, k in zip(ax[:4], [20, 60, 100, rank2]):
    show(tsvd(k), a, f"first {k} singular vectors", vmin=0.7, vmax=1.3)
ax[4].semilogy(ks, model_error, label="model error ‖m̂ − m_true‖")
ax[4].semilogy(ks, np.array(data_misfit) / len(d_noisy), label="data misfit χ²/n")
ax[4].set(xlabel="number of singular vectors kept, k"); ax[4].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''),

md(r"""
Adding singular vectors first sharpens the image and then destroys it: beyond about a hundred terms the data misfit keeps falling but the model error rises, because the remaining terms are dominated by noise divided by small singular values. Fitting the data as well as possible is the *wrong* goal. The truncation point $k$ is a choice, and a different noise realisation would move it.

### Regularisation

The remedy is to give up fitting the data *exactly* and instead minimise a compromise between data misfit and model complexity:

$$
\chi^2(\mathbf{m}) + \lambda\,\|\mathbf{m} - \mathbf{m}_0\|^2, \qquad
\chi^2 = \sum_k \frac{(d_k - (\mathbf{G}\mathbf{m})_k)^2}{\sigma_k^2},
$$

whose minimiser is the **damped least-squares** (Tikhonov) solution

$$
\hat{\mathbf{m}} = \mathbf{m}_0 + \big(\mathbf{G}^T\mathbf{G} + \lambda\sigma_d^2\,\mathbf{I}\big)^{-1}\mathbf{G}^T(\mathbf{d} - \mathbf{G}\mathbf{m}_0).
$$

The **damping parameter** $\lambda$ controls the compromise. As $\lambda \to 0$ we recover the unstable pseudo-inverse; as $\lambda \to \infty$ we get the reference model regardless of data. In between, the damping suppresses the components along small singular values (each is multiplied by $s_i^2/(s_i^2 + \lambda\sigma_d^2)$) and so kills the amplified noise, at the cost of also suppressing any *real* structure in those directions. Choosing $\lambda$ is a judgement. A common rule, the **discrepancy principle**, is to take the smallest damping for which the data are fitted to within their errors, $\chi^2 \approx n$.
"""),

code(r'''
def damped_ls(G, d, m0, sigma_d, lam):
    A = G.T @ G + lam * sigma_d**2 * np.eye(G.shape[1])
    return m0 + np.linalg.solve(A, G.T @ (d - G @ m0))

lams = np.logspace(-2, 5, 60)
chi2 = np.array([np.sum(((d_noisy - G2 @ damped_ls(G2, d_noisy, m0, sigma_d, l)) / sigma_d)**2) for l in lams])
norm = np.array([np.linalg.norm(damped_ls(G2, d_noisy, m0, sigma_d, l) - m0) for l in lams])

lam_star = lams[np.argmin(np.abs(chi2 - len(d_noisy)))]      # discrepancy principle
m_damped = damped_ls(G2, d_noisy, m0, sigma_d, lam_star)
print(f"discrepancy principle picks λ ≈ {lam_star:.2g}  (χ² = {np.sum(((d_noisy - G2 @ m_damped)/sigma_d)**2):.0f} for n = {len(d_noisy)} data)")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].loglog(chi2, norm, "o-", ms=3)
ax[0].plot(chi2[np.argmin(np.abs(chi2 - len(d_noisy)))], norm[np.argmin(np.abs(chi2 - len(d_noisy)))], "C3o", label="χ² ≈ n")
ax[0].axvline(len(d_noisy), color="k", lw=1, ls="--")
ax[0].set(xlabel="data misfit χ²", ylabel="model norm ‖m − m₀‖", title="trade-off curve"); ax[0].legend()
show(m_true, ax[1], "true model", vmin=0.7, vmax=1.3)
show(m_damped, ax[2], f"damped least squares, λ = {lam_star:.2g}", vmin=0.7, vmax=1.3)
plt.show()
'''),

md(r"""
The damped solution recovers the two anomalies, blurred and with reduced amplitude. The trade-off curve makes the compromise explicit: moving down it fits the data better with an ever larger, and eventually absurd, model. The knee of the curve, where the discrepancy principle lands, is the usual place to stop. Different choices of $\lambda$, or of what "complexity" means (the norm of the model, of its gradient, of its departure from some other reference), give different images from the same data. A tomographic image is not a photograph.
"""),

md(r"""
## Resolution: what does the estimate actually show?

Every linear estimate is a linear function of the data, $\hat{\mathbf{m}} - \mathbf{m}_0 = \mathbf{H}(\mathbf{d} - \mathbf{G}\mathbf{m}_0)$ for some matrix $\mathbf{H}$ (the pseudo-inverse, or the damped inverse above). For error-free data, $\mathbf{d} = \mathbf{G}\mathbf{m}_{\text{true}}$, and so

$$
\hat{\mathbf{m}} - \mathbf{m}_0 = \mathbf{R}\,(\mathbf{m}_{\text{true}} - \mathbf{m}_0), \qquad \mathbf{R} = \mathbf{H}\mathbf{G} .
$$

$\mathbf{R}$ is the **resolution matrix**. If it were the identity, the estimate would equal the truth. It never is, in an underdetermined problem: each row of $\mathbf{R}$ is a **resolving kernel** that says which weighted average of the true model the corresponding element of the estimate represents. The estimate is a *blurred* version of the truth, and $\mathbf{R}$ is the blur. This is computable, and it should be looked at.

A popular shortcut in tomography is the **checkerboard test**: put a checkerboard pattern through the whole procedure and see how much of it comes back. It amounts to looking at $\mathbf{R}$ applied to one particular model.
"""),

code(r'''
H = np.linalg.solve(G2.T @ G2 + lam_star * sigma_d**2 * np.eye(M), G2.T)     # the damped inverse
R = H @ G2                                                                    # resolution matrix

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, cell in zip(ax[:2], [(6, 6), (0, 11)]):
    j = cell[0] + N * cell[1]
    show(R[j], a, f"resolving kernel for cell {cell}")
    a.plot((cell[0] + 0.5) / N, (cell[1] + 0.5) / N, "k+", ms=12)
checker = np.where(((X * N // 2).astype(int) + (Y * N // 2).astype(int)) % 2 == 0, 0.2, -0.2).ravel()
show(m0 + R @ checker, ax[2], "checkerboard test (error-free)", vmin=0.7, vmax=1.3)
plt.show()
print(f"diagonal of R: mean {np.mean(np.diag(R)):.2f} (1 would be perfect resolution);  trace of R = {np.trace(R):.1f} 'resolved parameters' out of {M}")
'''),

md(r"""
An interior cell is resolved as an average over a blob of neighbouring cells, elongated along the dominant ray direction. A corner cell, crossed by few rays, is resolved much more poorly. The trace of $\mathbf{R}$ counts the number of independent parameters the data really constrain at this level of damping, and it is a fraction of the 144 nominal unknowns. The checkerboard comes back blurred and attenuated, and worse near the edges.

Two warnings. Resolution says nothing about noise: a cell can be well resolved and still badly estimated if the data are poor. And the checkerboard test only tests checkerboards; a feature of a different shape or scale may be recovered better or worse.
"""),

md(r"""
## The Bayesian view: regularisation is a prior

Notebook 7 showed that multiplying a likelihood by a prior gives a posterior. Take a Gaussian prior for the model, centred on $\mathbf{m}_0$ with covariance $\mathbf{C}_m$, and Gaussian data errors with covariance $\mathbf{C}_d = \sigma_d^2\mathbf{I}$. The posterior is Gaussian, with

$$
\mathbf{C}_{\text{post}} = \big(\mathbf{G}^T\mathbf{C}_d^{-1}\mathbf{G} + \mathbf{C}_m^{-1}\big)^{-1}, \qquad
\hat{\mathbf{m}} = \mathbf{m}_0 + \mathbf{C}_{\text{post}}\,\mathbf{G}^T\mathbf{C}_d^{-1}(\mathbf{d} - \mathbf{G}\mathbf{m}_0).
$$

With $\mathbf{C}_m = \sigma_m^2\mathbf{I}$ the posterior mean is *exactly* the damped least-squares solution with $\lambda = 1/\sigma_m^2$. Damping is a prior in disguise: the damping parameter states how large you believe model perturbations to be. This is clarifying in two ways. First, it says what regularisation *means*, and lets you build in better information, such as spatial smoothness through correlations in $\mathbf{C}_m$. Second, the posterior covariance provides uncertainties and, better still, **samples**: a set of models each of which is consistent with the data *and* the prior. Their spread is the honest answer to "what do we know?"

Here we use a prior with smooth spatial correlations (nearby cells are likely to have similar slowness) and draw samples from the prior and the posterior.
"""),

code(r'''
cells = np.column_stack([X.ravel(), Y.ravel()])
dist = np.linalg.norm(cells[:, None, :] - cells[None, :, :], axis=2)
sigma_m, ell = 0.2, 0.15                                            # prior amplitude and correlation length
C_m = sigma_m**2 * np.exp(-dist**2 / (2 * ell**2)) + 1e-8 * np.eye(M)

C_post = np.linalg.inv(G2.T @ G2 / sigma_d**2 + np.linalg.inv(C_m))
m_post = m0 + C_post @ G2.T @ (d_noisy - G2 @ m0) / sigma_d**2
std_post = np.sqrt(np.diag(C_post))

fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
for a in ax[0, :3]:
    show(rng.multivariate_normal(m0, C_m), a, "a prior sample", vmin=0.7, vmax=1.3)
show(m_true, ax[0, 3], "true model", vmin=0.7, vmax=1.3)
for a in ax[1, :2]:
    show(rng.multivariate_normal(m_post, C_post), a, "a posterior sample", vmin=0.7, vmax=1.3)
show(m_post, ax[1, 2], "posterior mean", vmin=0.7, vmax=1.3)
im = show(std_post, ax[1, 3], "posterior sd", vmin=0, vmax=sigma_m, cmap="viridis"); plt.colorbar(im, ax=ax[1, 3])
plt.show()
print(f"prior sd everywhere: {sigma_m};  posterior sd ranges from {std_post.min():.3f} (interior) to {std_post.max():.3f} (top and bottom edges)")
'''),

md(r"""
Prior samples are smooth random fields of the assumed amplitude and scale: what we thought plausible before looking at the data. Posterior samples all show the two anomalies, in roughly the right places, but differ in detail, particularly near the top and bottom edges where the fan rays are sparse and only the base rays pass: that is what the data leave undetermined. The posterior mean is the smooth compromise, and the standard deviation map shows where it can be trusted. Compare it with the resolving kernels above: the same geometry is at work.

The prior is doing real work here, and a different prior (larger amplitude, shorter correlation length) would give different posterior samples from the same data. That is not a defect of the method; it is the non-uniqueness of the inverse problem made explicit. The data alone do not determine the model, so something else must, and the Bayesian formulation forces you to say what.
"""),

md(r"""
## Optional: the same problem in function space with `pygeoinf`

The toy problem discretised the block into 64 cells, and the null space had at most 64 dimensions. Really the unknown is a *function* of position, and the null space is infinite-dimensional: there are infinitely many independent ways to change a function without changing a finite number of measurements. Working with functions directly needs a little more mathematics (Hilbert spaces of functions, and Gaussian measures on them, in place of vectors and covariance matrices), but the structure of the calculation is exactly as above.

The Python library [`pygeoinf`](https://github.com/da380/pygeoinf), developed in this department, provides that machinery. Below we infer a function on a circle from fifteen point measurements. Run the cell to install the library if it is missing (it needs Python 3.12 or later, which Colab provides).
"""),

code(r'''
try:
    import pygeoinf as inf
except ImportError:
    %pip install -q pygeoinf==1.8.9
    import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev

# The model space: functions on a circle, represented by their Fourier coefficients up to degree 128
X_space = Sobolev(128, 2.0, 0.1)
theta = X_space.points()                                          # angles at which functions are tabulated

# The prior: a Gaussian measure on the space whose samples are smooth functions with a chosen length scale
prior = X_space.heat_kernel_gaussian_measure(0.05)

# The forward operator: point measurements at 15 random angles, with independent errors of sd 0.1
obs = np.sort(rng.uniform(0, 2 * np.pi, 15))
A = X_space.point_evaluation_operator(obs)
data_errors = inf.GaussianMeasure.from_standard_deviation(A.codomain, 0.1)
problem = inf.LinearForwardProblem(A, data_error_measure=data_errors)

# A true model drawn from the prior, with synthetic data; then the Bayesian posterior
u_true, d = problem.synthetic_model_and_data(prior)
posterior = inf.LinearBayesianInversion(problem, prior).model_posterior_measure(d, inf.CholeskySolver())
u_mean = posterior.expectation
u_std = posterior.sample_pointwise_std(200)

fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for _ in range(3):
    ax[0].plot(theta, prior.sample(), lw=1)
ax[0].set(title="three samples from the prior", xlabel="angle θ", ylabel="u(θ)")
for _ in range(5):
    ax[1].plot(theta, posterior.sample(), lw=0.8, color="0.6")
ax[1].fill_between(theta, u_mean - 2*u_std, u_mean + 2*u_std, alpha=0.2, label="posterior mean ± 2 sd")
ax[1].plot(theta, u_mean, "C1", label="posterior mean")
ax[1].plot(theta, u_true, "k--", lw=1, label="true function")
ax[1].errorbar(obs, d, yerr=0.1, fmt="C3o", ms=4, label="data")
ax[1].set(title="posterior: samples, mean, and the truth", xlabel="angle θ"); ax[1].legend(fontsize=8)
plt.show()
'''),

md(r"""
Every grey curve fits the fifteen data to within their errors and is a plausible smooth function; between the data points they disagree, and the shaded band quantifies by how much. Note that the unknown here has (in principle) infinitely many degrees of freedom and the data only fifteen, yet the calculation is no harder than the matrix version: the library handles the function-space bookkeeping, and the six lines that define the problem and solve it would be the same for a function on a sphere with data from a seismic network.

The tomographic version needs one change: replace point evaluations by integrals along paths. Below, three sources and four receivers on the circle give twelve chords, and the data are the integrals of the unknown function along them.
"""),

code(r'''
paths = X_space.random_source_receiver_paths(3, 4)               # every source to every receiver: 12 chords
T = X_space.path_average_operator(paths)
tomo = inf.LinearForwardProblem(T, data_error_measure=inf.GaussianMeasure.from_standard_deviation(T.codomain, 0.05))
u_true2, d2 = tomo.synthetic_model_and_data(prior)
post2 = inf.LinearBayesianInversion(tomo, prior).model_posterior_measure(d2, inf.CholeskySolver())
u2_mean, u2_std = post2.expectation, post2.sample_pointwise_std(200)

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(theta, u2_mean - 2*u2_std, u2_mean + 2*u2_std, alpha=0.2, label="posterior mean ± 2 sd")
ax.plot(theta, u2_mean, "C1", label="posterior mean")
ax.plot(theta, u_true2, "k--", lw=1, label="true function")
ax.set(xlabel="angle θ", ylabel="u(θ)", title="12 path integrals: what the data can and cannot say"); ax.legend(fontsize=8)
plt.show()
print(f"chi-squared of the true model against the data: {tomo.chi_squared(u_true2, d2):.1f} for {len(paths)} data")
'''),

md(r"""
Twelve integrals constrain a function only in an average sense, and the posterior band is correspondingly wide; the mean captures the broad shape and no more. That is the honest outcome, and it is what the data support.
"""),

md(r"""
## Summary

- In an underdetermined problem the data fix only a few combinations of the model parameters; the null space, the set of models producing no data, contains everything else. Exact data do not help.
- The minimum-norm solution is the simplest data-fitting model in one specific sense. It is a choice.
- Small singular values amplify noise; damping (regularisation) trades data fit for model simplicity, and the trade-off curve makes the choice visible.
- The resolution matrix says which average of the truth each element of the estimate represents. Look at it.
- Regularisation is a prior. The Bayesian posterior gives uncertainties and samples that show what the data leave open.

**Further reading.** R. L. Parker, *Geophysical Inverse Theory* (Princeton), is the classic account of these ideas, and A. Tarantola, *Inverse Problem Theory* (SIAM, free online), the standard reference for the Bayesian view. The `pygeoinf` documentation at https://pygeoinf.readthedocs.io has worked examples on the line, circle, plane and sphere.
"""),

md(r"""
## Exercises
"""),

exercise(1, r"""
Return to the 24-ray experiment. Add a single diagonal ray from the bottom-left corner to the top-right corner. By how much does the rank increase? Now add the *other* diagonal. Then add rays from the bottom-left corner to each of the $N$ points on the right edge. Track the rank and the smallest non-zero singular value as rays are added, and explain why extra rays can raise the rank without making the problem much better determined in practice.
"""),
scratch(),
solution(r"""
```python
extra = [[((0, 0), (1, 1))],
         [((0, 0), (1, 1)), ((0, 1), (1, 0))],
         [((0, 0), (1, 1)), ((0, 1), (1, 0))] + [((0, 0), (1, b)) for b in edge]]
for e in extra:
    Ge = ray_matrix(rays + e, N)
    se = np.linalg.svd(Ge, compute_uv=False)
    r = np.linalg.matrix_rank(Ge)
    print(f"{len(rays) + len(e):3d} rays: rank {r:2d}, smallest non-zero singular value {se[r-1]:.3f}, largest {se[0]:.3f}")
```

Each diagonal adds one to the rank, and it is a healthy addition: the new singular value is as large as the existing ones, because a diagonal crosses many cells in a combination the row and column sums cannot form. The corner fan of twelve rays adds eleven more to the rank, but the smallest singular value drops by a factor of ten: those rays all pass through the same corner cells and differ from one another only slightly, so the new directions they constrain are constrained only weakly. Rank counts independent directions without regard to how *well* they are determined; the singular values are the honest measure, and they show that a handful of extra rays leaves the problem effectively underdetermined.
"""),

exercise(2, r"""
Repeat the damped least-squares inversion with the travel-time errors doubled (`sigma_d = 0.01`; regenerate the noisy data). Find the damping picked by the discrepancy principle, plot the trade-off curve and the recovered model, and compare the trace of the resolution matrix with the earlier value. What has the extra noise cost?
"""),
scratch(),
solution(r"""
```python
sd2 = 0.01
d2n = G2 @ m_true + stats.norm(0, sd2).rvs(len(rays2), random_state=rng)
chi2b = np.array([np.sum(((d2n - G2 @ damped_ls(G2, d2n, m0, sd2, l)) / sd2)**2) for l in lams])
lam2 = lams[np.argmin(np.abs(chi2b - len(d2n)))]
mb = damped_ls(G2, d2n, m0, sd2, lam2)
Rb = np.linalg.solve(G2.T @ G2 + lam2 * sd2**2 * np.eye(M), G2.T) @ G2
print(f"σ_d = {sd2}: λ = {lam2:.2g}, trace of R = {np.trace(Rb):.1f} (was {np.trace(R):.1f} with σ_d = {sigma_d})")
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
show(m_true, ax[0], "true model", vmin=0.7, vmax=1.3); show(mb, ax[1], f"damped LS, σ_d = {sd2}", vmin=0.7, vmax=1.3); plt.show()
```

Noisier data require heavier effective damping, the number of resolved parameters falls, and the recovered anomalies are blurrier and weaker. Resolution is not a property of the ray geometry alone; it depends on how good the data are.
"""),

exercise(3, r"""
In the `pygeoinf` example, change the prior length scale (the argument of `heat_kernel_gaussian_measure`, try 0.01 and 0.2) and the number of observation points (5 and 60), keeping the same true function where possible. How do the posterior mean and the width of the posterior band respond? When does the prior matter most?
"""),
scratch(),
solution(r"""
```python
fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
for row, scale in zip(ax, [0.01, 0.2]):
    pr = X_space.heat_kernel_gaussian_measure(scale)
    for a, n_obs in zip(row, [5, 60]):
        pts = np.sort(rng.uniform(0, 2*np.pi, n_obs))
        Ao = X_space.point_evaluation_operator(pts)
        pb = inf.LinearForwardProblem(Ao, data_error_measure=inf.GaussianMeasure.from_standard_deviation(Ao.codomain, 0.1))
        dd = pb.synthetic_data(u_true)                        # same true function, new data
        po = inf.LinearBayesianInversion(pb, pr).model_posterior_measure(dd, inf.CholeskySolver())
        mu, sd = po.expectation, po.sample_pointwise_std(100)
        a.fill_between(theta, mu - 2*sd, mu + 2*sd, alpha=0.2); a.plot(theta, mu, "C1"); a.plot(theta, u_true, "k--", lw=1)
        a.plot(pts, dd, "C3o", ms=3); a.set(title=f"prior scale {scale}, {n_obs} data")
plt.show()
```

A short prior length scale allows rapid wiggles, so between sparse data points the band is wide and the mean returns to the prior mean of zero; a long length scale forces smooth interpolation with a narrow band, which is only trustworthy if the truth really is that smooth. With 60 data the two priors give similar answers because the data dominate. The prior matters most where the data are sparse, which is exactly where the inverse problem is most underdetermined.
"""),
]

build(F, cells)
