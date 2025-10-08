## Reproducibility

Compute and save the log-likelihood and f1 score of a model on TriviaQA data. "model_name" specifies the name of the model
```{python}
python triviaqa_f1.py --model_name facebook/opt-1.3b
python triviaqa_likelihood.py --model_name facebook/opt-1.3b
```
"read_triviaqa_result.ipynb" contains code to compute the estimator and construct confidence interval using saved results.
