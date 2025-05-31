## Reproducibility

Compute and save the log-likelihood of variants of GPT2 model on WikiText data. "model_name" specifies the GPT2 model size, "flag" controls quantization.
```{python}
python gpt_loglikelihood.py --flag 0 --model_name gpt2
```
"estimation.ipynb" contains code to compute the estimator and construct confidence interval using saved results.
