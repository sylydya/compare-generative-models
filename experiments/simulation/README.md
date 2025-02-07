## Reproducibility

Compute estimators of difference of KL divergence using our methods with known $g$ or with learnd $g$ using auto-encoder
```{python}
python simu_ours.py
python simu_ours_ae.py
```
Compute estimators of difference of KL divergence using k nearest neighbour estimator and use Subsampling or Adaptive HulC to construct confidence intervals
```{python}
python simu_KL_hulc.py
python simu_KL_subsampling.py
```
Compute estimators of difference of W2 distance using plug-in estimator and use Subsampling or Adaptive HulC to construct confidence intervals
```{python}
python simu_W2_hulc.py
python simu_W2_subsampling.py
```