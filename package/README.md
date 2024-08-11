# Python package <!-- omit in toc -->

> [!NOTE]
> See also [RTDL](https://github.com/yandex-research/rtdl)
> -- other projects on tabular deep learning.

This package provides the officially recommended
implementation of the paper "On Embeddings for Numerical Features in Tabular Deep Learning".

<details>
<summary><i>This package VS The original implementation</i></summary>

"Original implementation" is the code in `bin/` and `lib/`
used to obtain numbers reported in the paper.

- **This package is recommended over the original implementation**:
  the package is significanty simpler
  while being fully consistent with the original code
  (with one minor exception: there is one accidental divergence
  of the original code from the paper, which is now fixed in the package)
- Strictly speaking, the package may have
  small technical divergences from the original code.
  Just in case, they are marked
  with `# NOTE[DIFF]` comments in the source code of this package.
  Any divergence from the original implementation without the `# NOTE[DIFF]` comment
  is considered to be a bug.

</details>


> [!IMPORTANT]
> For a long time, in the main branch of the
> [RTDL](https://github.com/yandex-research/rtdl) project,
> there was an *unfinished* implementation of this paper with many unresolved issues.
> It is *highly* recommended to switch to this package.

---

- [Installation](#installation)
- [Usage](#usage)
- [End-to-end examples](#end-to-end-examples)
- [Practical notes](#practical-notes)
- [API](#api)
- [Development](#development)

# Installation

> [!NOTE]
> If you are *not* going to use
> the decision-tree-based bin computation (`compute_bins(..., tree_kwargs={...})`),
> then you can omit the installation of `scikit-learn`.

*(RTDL ~ **R**esearch on **T**abular **D**eep **L**earning)*

```
pip install rtdl_num_embeddings
pip install "scikit-learn>=1.0,<2"
```

# Usage

> [!IMPORTANT]
> It is recommended to first read the TL;DR of the paper:
> [link](../README.md#tldr)

Let's consider a toy tabular data problem where objects are represented by three
continuous features
(for simplicity, other feature types are omitted,
but they are covered in the end-to-end example):

<!-- test main -->
```python
# NOTE: all code snippets can be copied and executed as-is.
import torch
import torch.nn as nn
from rtdl_num_embeddings import (
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
    compute_bins,
)
# NOTE: pip install rtdl_revisiting_models
from rtdl_revisiting_models import MLP

batch_size = 256
n_cont_features = 3
x = torch.randn(batch_size, n_cont_features)
```

This is how a vanilla MLP **without embeddings** would look like:

<!-- test main -->
```python
mlp_config = {
    'd_out': 1,  # For example, a single regression task.
    'n_blocks': 2,
    'd_block': 256,
    'dropout': 0.1,
}
model = MLP(d_in=n_cont_features, **mlp_config)
y_pred = model(x)
```

And this is how MLP **with embeddings for continuous features** can be created:

<!-- test main -->
```python
d_embedding = 24
m_cont_embeddings = PeriodicEmbeddings(n_cont_features, lite=False)
model_with_embeddings = nn.Sequential(
    # Input shape: (batch_size, n_cont_features)

    m_cont_embeddings,
    # After embeddings: (batch_size, n_cont_features, d_embedding)

    # NOTE: `nn.Flatten` is not needed for Transformer-like architectures.
    nn.Flatten(),
    # After flattening: (batch_size, n_cont_features * d_embedding)

    MLP(d_in=n_cont_features * d_embedding, **mlp_config)
    # The final shape: (batch_size, d_out)
)
# The usage is the same as for the model without embeddings:
y_pred = model_with_embeddings(x)
```

In other words, the whole paper is about the fact that having such a thing as
`m_cont_embeddings` can (significantly) improve the downstream performance.

**The paper showcases three types of such embeddings**:
- [Simple](#simple-embeddings)
- [Periodic](#simple-embeddings)
- [Piecewise-linear](#piecewise-linear-encoding--embeddings)

## Simple embeddings<!-- omit in toc -->

*(Decribed in Section 3.4 in the paper)*

| Name | Definition for a single feature | How to create               |
| :--- | :------------------------------ | :-------------------------- |
| `LR` | `ReLU(Linear(x_i))`             | `LinearReLUEmbeddings(...)` |

In the above table:
- L ~ Linear, R ~ ReLU.
- `x_i` is the i-th scalar continuous feature

**Hyperparameters**

- The default value of `d_embedding` is set with the MLP backbone in mind.
  Typically, for Transformer-like backbones, the embedding size is larger.
- For MLP, on most tasks (at least on non-small tasks),
  tuning `d_embedding` will not have much effect.
- See other notes on hyperparameters in ["Practical notes"](#practical-notes).

<!-- test main _ -->
```python
# MLP-LR
d_embedding = 32
model = nn.Sequential(
    LinearReLUEmbeddings(n_cont_features, d_embedding),
    nn.Flatten(),
    MLP(d_in=n_cont_features * d_embedding, **mlp_config)
)
y_pred = model(x)
```

<details>
<summary>Advanced example</summary>

To further illustrate the overall idea, let's consider a more advanced example,
where embeddings consist of three steps:
1. First, each feature is embedded linearly.
2. Then, ReLU is applied.
   At this point, the embedding is equivalent to `LinearReLUEmbeddings`.
3. Finally, feature embeddings are project to a lower dimension,
   where *separete* (i.e. non-shared) linear projections are learned
   for all feature.

<!-- test main _ -->
```python
# NOTE: pip install delu
import delu
from rtdl_revisiting_models import LinearEmbeddings

m_embeddings = nn.Sequential(
    LinearEmbeddings(n_cont_features, 48),
    nn.ReLU(),
    delu.nn.NLinear(n_cont_features, 48, 8)
)
model = nn.Sequential(
    m_embeddings,
    nn.Flatten(),
    MLP(d_in=n_cont_features * 8, **mlp_config)
)
y_pred = model(x)
```

</details>

## Periodic embeddings<!-- omit in toc -->

*(Decribed in Section 3.3 in the paper)*

| Name        | Definition for a single feature                                            | How to create                                           |
| :---------- | :------------------------------------------------------------------------- | :------------------------------------------------------ |
| `PLR`       | `ReLU(Linear(Periodic(x_i)))`                                              | `PeriodicEmbeddings(..., lite=False)`                   |
| `PLR(lite)` | `ReLU(Linear(Periodic(x_i)))` <br> *(`Linear` is shared between features)* | `PeriodicEmbeddings(..., lite=True)`                    |
| `PL`        | `Linear(Periodic(x_i))`                                                    | `PeriodicEmbeddings(..., activation=False, lite=False)` |

In the above table:
- P ~ Periodic, L ~ Linear, R ~ ReLU.
- `x_i` is the i-th scalar continuous feature
- `Periodic(x_i) = concat[cos(v_i), sin(v_i)]`, where, **following Section 3.3**:
  - `v_i = [2 * pi * c_1 * x_i, ..., 2 * pi * c_k * x_i]`
    where `k` is set with the `n_frequencies` hyperparameter.
  - The `frequency_init_scale` hyperparameter is the initialization scale for the `c_i` coefficients.
- `lite` is a new option introduced and used in a *different* paper ([this one](https://github.com/yandex-research/tabular-dl-tabr/)).
  On some tasks, it allows making the `PLR` embedding significantly more lightweight
  at the cost of non-critical performance loss.

**Hyperparameters**

- `n_frequencies` and `frequency_init_scale` are commented above.
- <details><summary><b>How to tune the <code>frequency_init_scale</code> hyperparameter</b></summary>

  **Prioritize testing smaller values, because they are safer:**
  - Larger-than-the-optimal value can lead to terrible performance.
  - Smaller-than-the-optimal value will still yield decent performance.

  Some approximate numbers:
  - for 30% of tasks, the optimal `frequency_init_scale` is less than 0.05.
  - for 50% of tasks, the optimal `frequency_init_scale` is less than 0.2.
  - for 80% of tasks, the optimal `frequency_init_scale` is less than 1.0.
  - for 90% of tasks, the optimal `frequency_init_scale` is less than 5.0.

  If you want to test larger values,
  make sure that you have enough hyperparameter tuning budget
  (e.g. at least 100 trials of the TPE Optuna sampler, as in the paper).

  </details>

- The default value of `d_embedding` is set with the MLP backbone in mind.
  Typically, for Transformer-like backbones, the embedding size is larger.
- See other notes on hyperparameters in ["Practical notes"](#practical-notes).

<!-- test main _ -->
```python
# Example: MLP-PLR
d_embedding = 24
model = nn.Sequential(
    PeriodicEmbeddings(n_cont_features, d_embedding, lite=False),
    nn.Flatten(),
    MLP(d_in=n_cont_features * d_embedding, **mlp_config)
)
y_pred = model(x)
```

## Piecewise-linear encoding & embeddings<!-- omit in toc -->

*(Decribed in Section 3.2 in the paper)*

<img src="piecewise-linear-encoding.png" width=40%>

| Name                               | Definition for a single feature | How to create                                       |
| :--------------------------------- | :------------------------------ | :-------------------------------------------------- |
| `Q`/`T` (only for MLP-like models) | `ple(x_i)`                      | `PiecewiseLinearEncoding(bins)`                     |
| `QL`/`TL`                          | `Linear(ple(x_i))`              | `PiecewiseLinearEmbeddings(bins, activation=False)` |
| `QLR` / `TLR`                      | `ReLU(Linear(ple(x_i)))`        | `PiecewiseLinearEmbeddings(bins, activation=True)`  |

In the above table:
- Q ~ quantiles-based bins, T ~ tree-based bins, L ~ Linear, R ~ ReLU.
- `x_i` is the i-th scalar continuous feature.
- `ple` stands for "Piecewise-linear encoding".

**Notes**

- In the table above, there are *two* distinct classes:
  `PiecewiseLinearEncoding` and `PiecewiseLinearEmbeddings`.
- The output of `PiecewiseLinearEncoding` has the shape `(*batch_dims, d_encoding)`,
  where `d_encoding` equals the total number of bins of all features.
  This variation of piecewise-linear representations
  without end-to-end trainable parameters is suitable only for MLP-like models.
- By contrast, `PiecewiseLinearEmbeddings` is similar to all other classes of
  this package and its output has the shape `(*batch_dims, n_features, d_embedding)`.

**Hyperparameters**

- For `PiecewiseLinearEmbeddings`,
  possible starting points are `d_embedding=8, activation=False`
  or `d_embedding=24, activation=True`.
- See other notes on hyperparameters in ["Practical notes"](#practical-notes).

<!-- test main _ -->
```python
X_train = torch.randn(10000, n_cont_features)
Y_train = torch.randn(len(X_train))  # Regression.

# (Q) Quantile-based bins.
bins = compute_bins(X_train)

# (T) Target-aware tree-based bins.
#     They are extracted from splitting nodes
#     of feature-wise decision trees.
bins = compute_bins(
    X_train,
    # NOTE: requires scikit-learn>=1.0 to be installed.
    tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    y=Y_train,
    regression=True,
)

# MLP-Q / MLP-T
model = nn.Sequential(
    PiecewiseLinearEncoding(bins),
    nn.Flatten(),
    MLP(d_in=sum(len(b) - 1 for b in bins), **mlp_config)
)
y_pred = model(x)

# MLP-QLR / MLP-TLR
d_embedding = 24
model = nn.Sequential(
    PiecewiseLinearEmbeddings(bins, d_embedding, activation=True),
    nn.Flatten(),
    MLP(d_in=n_cont_features * d_embedding, **mlp_config)
)
y_pred = model(x)
```

# End-to-end examples

See [this Jupyter notebook](./example.ipynb) (Colab link inside).

# Practical notes

**General comments**

- **Embeddings for continuous features are applicable to most tabular DL models**
  and often lead to better performance.
  On some problems, embeddings can lead to truly significant improvements.
- As of 2022-2023, **MLP with embeddings is a reasonable modern baseline**
  in terms of both task performance and efficiency.
  Depending on the task and embeddings, it can perform on par or even better than
  FT-Transformer, while being significantly more efficient.
- Despite the formal overhead in terms of parameter count,
  **embeddings are perfectly affordable in many cases**.
  That said, on big enough datasets and/or with large enough number of features and/or
  with strict enough latency requirements,
  the new overhead associated with embeddings may become an issue.

**Practical overview of the embeddings**

*(this section assumes MLP as the backbone)*

`LinearReLUEmbeddings`:
- A lightweight embedding falling into the "low risk & (usually) low reward" category.
- A good choice for a quick start on a new problem, especially if 
  this is your first time working with embeddings.

`PeriodicEmbeddings`:
- Demonstrated the best average performance in the paper.
- Often, the "lite" version `PeriodicEmbeddings(..., lite=True)` is a good
  starting point in terms of the balance between task performance and efficiency.
- So, in practice, a possible strategy is to start with `lite=True`,
  tune hyperparameters if needed, and then try `lite=False`.

`PiecewiseLinearEncoding` & `PiecewiseLinearEmbeddings`:
- Why trying this if the periodic embeddings are better on average?
  There is no single reason, rather a range of small things that
  can make piecewise-linear representations preferrable in some cases:
  - To start with, they just work well on some problems.
  - They make a model less sensitive to feature preprocessing:
    (1) standardization (`sklearn.preprocessing.StandardScaler`)
        becomes unneeded.
    (2) quantile transformation can still be useful,
        but may become less impactful on some problems.
  - They can occasionally make a model more robust to outliers in the training data.
  - They are simpler to understand and reason about. In particular,
    `PiecewiseLinearEmbeddings` can be seen as a collection of bin
    embeddings that are aggregated based on input feature values.
  - The quantile-based bins are somewhat easy to use due to just one hyperparameter
    (good defaults for tree-based bins may exist as well,
    but there were no attempts to find them; perhaps, the published tuned configurations
    for different datasets contain the answer).
- Regarding the drawbacks:
  - In some setups, they can be less convenient to use
    because of the additional bin computation step.

**Hyperparameters**

> [!NOTE]
> It is possible to explore tuned hyperparameters
> for the models and datasets used in the paper as explained here:
> [link](../README.md#how-to-explore-metrics-and-hyperparameters).

- The default hyperparameters are set with the MLP-like backbones in mind and
  with "low risk" (not the "high reward") as the priority.
  For Transformer-like models, one may need to (significantly) increase `d_embedding`.
- Tuning hyperparameters of the periodic embeddings can require special considerations
  as described in the [corresponding usage section](#periodic-embeddings).
- For MLP-like models with embeddings ending with a linear layer `L`
  (e.g. `PL`, `QL`, `TL`),
  a possible starting point is to set `d_embedding` to a smaller-than-default value (e.g. `8` or `16`).
- In the paper, for hyperparameter tuning, the
  [TPE sampler from Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
  was used with `study.optimize(..., n_trials=100)` (sometimes, `n_trials=50`).
- The hyperparamer tuning spaces can be found in the appendix of the paper
  and in `exp/**/*tuning.toml` files
  (for the `frequency_init_scale` hyperparameter of `PeriodicEmbeddings`,
  the upper bound can often be safely reduced to `10.0` instead of `100.0`).

**Tips**

- To improve efficiency, it is possible to embed only a subset of features.
- The biggest wins come from embedding *important, but "problematic"* features.
  Intuitively, "problematic features" are the ones that are hard to process
  for a given model and prevent it from achieving better results.
  (for example, features with irregular joint distributions
  with other features and labels may be such "problematic features").
- It is possible to combine embeddings
  and apply different embeddings to different features.
- The proposed embeddings are relevant only for continuous features,
  so they should not be used for embedding binary or categorical features.
- If an embedding ends with a linear layer (`PL`, `QL`, `TL`, etc.) and its output
  is passed to MLP, then that linear layer can be fused with the first linear layer of
  MLP after the training (sometimes, it can lead to better efficiency).
- (a bonus tip for those who read such long documents until the end)
  On some problems, MLP-L
  (that is, MLP with `rtdl_revisiting_models.LinearEmbeddings` -- the simplest possible
  linear embeddings from a different package) performs better than MLP.
  Combined with one of the bullets above, it means that, on some problems,
  one can train MLP-L and transform it to a simple embedding-free MLP after the training.

# API

To explore the available API and docstrings, open the source file and:
- On GitHub, use the Symbols panel.
- In VSCode, use the [Outline view](https://code.visualstudio.com/docs/getstarted/userinterface#_outline-view).
- Check the `__all__` variable.

# Development

<details>

Set up the environment (replace `micromamba` with `conda` or `mamba` if needed):
```
micromamba create -f environment-package.yaml
```

Check out the available commands in the [Makefile](./Makefile).
In particular, use this command before committing:
```
make pre-commit
```

Publish the package to PyPI (requires PyPI account & configuration):
```
flit publish
```
</details>

