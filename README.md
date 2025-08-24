# Ensemble-Based Survival Models with the Self-Attended Beran Estimator Predictions

Survival analysis predicts the time until an event of interest, such as failure or death, but faces challenges due to censored data, where some events remain unobserved. Ensemble-based models, like random survival forests and gradient boosting, are widely used but can produce unstable predictions due to variations in bootstrap samples. To address this, we propose SurvBESA (Survival Beran Estimators Self-Attended), a novel ensemble model that combines Beran estimators with a self-attention mechanism. Unlike traditional methods, SurvBESA applies self-attention to predicted survival functions, smoothing out noise by adjusting each survival function based on its similarity to neighboring survival functions. We also explore a special case using Huber's contamination model to define attention weights, simplifying training to a quadratic or linear optimization problem. Numerical experiments show that SurvBESA outperforms state-of-the-art models. The implementation of SurvBESA is publicly available.

## Contents


- `source` – Source files of the Beran model (beran.py) and the SurvBESA model (ensemble_beran.py)
