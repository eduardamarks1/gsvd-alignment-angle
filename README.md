=

# GSVD for Geometry-Grounded Dataset Comparison: An Alignment Angle Is All You Need

This repository contains experiments for **geometry-grounded dataset comparison** using the **Generalized Singular Value Decomposition (GSVD)**. The main goal is to compare two datasets at the **sample level**, rather than assigning a single global similarity score to the entire pair.

Our central object is the co-span relation

\[
Ax = By = z,
\]

which allows us to study vectors jointly explained by two datasets. From this relation, we define an interpretable **alignment angle** \(\theta(z)\) that indicates whether a sample is more naturally aligned with dataset \(A\), dataset \(B\), or shared by both.

---

## Main Idea

Many dataset comparison methods output a single number for a pair of datasets. While useful as a coarse summary, that does not explain:

- which specific samples drive similarity,
- which directions are shared,
- and which structures are dataset-specific.

This repository explores a different perspective:

- **small** \(\theta(z)\): sample is more aligned with dataset **A**
- **large** \(\theta(z)\): sample is more aligned with dataset **B**
- \(\theta(z) \approx \pi/4\): sample lies in a **shared or ambiguous region**

The same GSVD frame also reveals representative extreme directions, making the geometry interpretable.

---

## Method Overview

Given two matrices \(A\) and \(B\), we use the GSVD decomposition

\[
A = H C U, \qquad B = H S V,
\qquad C^\top C + S^\top S = I,
\]

to construct a coordinate system adapted to the joint geometry of the two datasets.

For a feasible vector \(z \in \operatorname{col}(A) \cap \operatorname{col}(B)\), we define the alignment angle by comparing the minimum-norm coefficients associated with \(A\) and \(B\):

\[
\theta(z) = \arctan\left(\frac{\|x\|_2}{\|y\|_2}\right).
\]

In the GSVD frame, this becomes

\[
\theta(z)
=
\arctan\left(
\frac{\|C^\dagger c(z)\|_2}{\|S^\dagger c(z)\|_2}
\right),
\qquad c(z)=H^\dagger z.
\]

This provides a simple and interpretable per-sample score.

---

## Information-Geometric View

The alignment angle naturally induces a Bernoulli posterior:

\[
P(A \mid \theta)=\cos^2\theta,
\qquad
P(B \mid \theta)=\sin^2\theta.
\]

This allows us to study the geometry of sample-level and distribution-level comparisons using Fisher--Rao distances.

In this view:

- angles near \(0\) or \(\pi/2\) correspond to **confident regions**
- angles near \(\pi/4\) correspond to **maximum ambiguity**
- overlap of angle histograms indicates **shared geometry**

---

## Experiments

The current experiments focus on **MNIST digit pairs**.

Each digit class is treated as a dataset in a shared ambient space after vectorization and mean-centering. We then:

1. build matrices \(A\) and \(B\) from two digit classes,
2. compute the GSVD-based joint frame,
3. score test samples using \(\theta(z)\),
4. analyze angle distributions,
5. inspect representative extreme directions.

Example observations:

- pairs such as **1 vs 5** show clearer separation in angle space,
- pairs such as **4 vs 9** show more overlap and stronger ambiguity.

The purpose is **not** state-of-the-art classification, but rather an interpretable diagnostic of relative dataset geometry.

---

## What This Repository Contains

Possible contents of the repository include:

- GSVD preprocessing code
- sample scoring with the alignment angle
- histogram and distribution plots
- reconstruction of representative GSVD directions
- experiment scripts for MNIST digit-pair comparisons

---

## Authors and Acknowledgments

This work was developed by:

- Arthur Sobrinho
- Eduarda Marques
- João Paixão
- Daniel S. Menasche
- Heudson Mirandola

We also thank **Júlia Motta** for her important contribution to the experimental pipeline, especially in organizing the outputs of the Julia GSVD implementation.

The Julia implementation does not naturally return the GSVD factors in the same organization adopted in our article. To match our formulation, it was necessary to apply permutations to the block structures of the matrices originally denoted here as **D1** and **D2** (corresponding to **C** and **S** in the paper), followed by a reordering step so that the values appearing in the pseudoinverses are arranged consistently with the alignment-angle interpretation, increasing from \(0^\circ\) to \(90^\circ\).

Some of the ideas that guided this organization were inspired by **Graphical Linear Algebra (GLA)**, particularly in thinking about block structure, composition, and the interpretation of the resulting operators.

### Naming note

In the current experiments, we still refer to the GSVD diagonal factors as **D1** and **D2**, which was our original notation. We plan to rename them to **C** and **S** in the codebase as well, to make the implementation consistent with the notation used in the article and improve clarity for future readers.
