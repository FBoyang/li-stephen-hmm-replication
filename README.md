# Li-Stephen HMM Replication

This is a Python replication of the **Li-Stephens Hidden Markov Model (HMM)**. This algorithm is widely used in computational genomics for genotype imputation, phasing, and modeling linkage disequilibrium.

## Background

The algorithm implementation is based on the seminal original paper:
> **Modeling Linkage Disequilibrium and Identifying Recombination Hotspots Using Single-Nucleotide Polymorphism Data**  
> Li, Na, and Matthew Stephens. *Genetics* 165.4 (2003): 2213-2233.  
> DOI: [10.1093/genetics/165.4.2213](https://doi.org/10.1093/genetics/165.4.2213)

For more detailed mathematical descriptions and notes regarding this specific implementation strategy, please refer to the [full description provided in Li-stephenHMM.pdf](./Li-stephenHMM.pdf).

## Repository Structure

- `hmm_valid.py`: The core algorithm implementation and analysis script. It contains functions for:
  - Emission/Transition Probability evaluations (e.g., `gammax`).
  - Forward-Backward recursion variants (with numerically stable log-space algorithms).
  - Model Evidence and Posterior evaluation routines.
- `805_SNP_1000G_real.hapt`: Real genotype/haplotype reference panel data.
- `805_SNP_1000G_real.hapt.train`: Training subset of the dataset.
- `805_SNP_1000G_real.hapt.test`: Testing subset of the dataset for validation and imputation testing.
- `Li-stephenHMM.pdf`: Background math logic underlying this replica.
- `805_SNP.map` (and others implicitly expected by the script but you should place in `./` depending on the block toggled): Genetic distance map mapping physical position to genetic map position.

## Dependencies

You will need Python 3 installed. The script depends heavily on vectorized operations and `@jit` compiled routines to accelerate the traditionally slow HMM recursions.
Install dependencies via pip:

```bash
pip install numpy pandas scipy numba tqdm
```

## Running the Code

The main script is ready to run and will trigger the analytical blocks defined at the bottom.
To execute it, run:

```bash
python3 hmm_valid.py
```

By default, the script loads training data, testing data, and the genetic map from the current working directory (`./`), and kicks off tasks defined in `main()` such as log-likelihood evaluation and parameter estimation testing.

You can modify boolean flags in `main()` (like `impute_FLAG`, `test_loglike`, `theta_learning`) to alter which analysis routines are executed.
