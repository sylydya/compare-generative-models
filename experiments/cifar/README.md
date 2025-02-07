## Reproducibility

Compute and save the Jacobian log determinant, "S" specifies the number of denoising steps, "skip" specifies the methods to select the subsequence of times steps
```{python}
python cifar_ddim_kl.py --S 50 --skip "quad"
python cifar_ddim_kl.py --S 100 --skip "quad"
```
"estimation.ipynb" contains code to compute the estimator and construct confidence interval using saved results.
