


**NeuralNetworkDTMC** is a hybrid sequential modeling framework that combines
**Discrete-Time Markov Chains (DTMCs)** with **neural regression and Gaussian decoding**, aiming to improve predictive accuracy while preserving interpretability.

Unlike end-to-end sequence models (e.g. RNNs), NeuralDTMC explicitly separates:

1. **State space construction** (discretization / binning)
2. **Transition modeling** (DTMC)
3. **Prediction decoding** (argmax vs expectation)

This design allows fine-grained evaluation of probabilistic quality, numerical accuracy, and calibration.




## Motivation

Traditional DTMCs estimate transition probabilities purely from empirical frequencies.
While simple and interpretable, they suffer from:

* Sparse transitions
* Poor generalization
* High numerical prediction error when states are ordered

NeuralDTMC addresses this by:

* Learning a **continuous regression model** 
* Modeling uncertainty via **Gaussian residuals**
* Decoding the resulting distribution into a **DTMC transition matrix**



## Evaluation Setup

* Same state space for all models
* Same validation split
* Metrics include:

  * **NLL**, **Brier score**
  * **Top-k accuracy** (argmax decoding)
  * **RMSE / MAE** (expectation decoding)
  * **Tolerance accuracy**
  * **Entropy & ECE** (calibration)



## Results Summary

Below is a representative comparison between:

* **Traditional frequency-based DTMC**
* **NeuralDTMC (NN + Gaussian decoding)**

###  Classification-style performance

| Metric         | Traditional DTMC | NeuralDTMC | Improvement |
| -------------- | ---------------- | ---------- | ----------- |
| Top-1 accuracy | 0.138            | **0.169**  | **+22%**    |
| Top-3 accuracy | 0.192            | **0.254**  | **+32%**    |
| Top-5 accuracy | 0.223            | **0.300**  | **+35%**    |




###  Numerical prediction accuracy

| Metric               | Traditional DTMC | NeuralDTMC | Improvement |         
| -------------------- | ---------------- | ---------- | ----------- | 
| MAE                  | 843.7            | **659.0**  | **−22%**    |         
| RMSE                 | 1054.9           | **990.7**  | **−6%**     |          





###  Calibration & uncertainty

| Metric       | Traditional DTMC | NeuralDTMC |
| ------------ | ---------------- | ---------- |
| Mean entropy | 4.82             | **3.10**   |
| ECE (Top-1)  | **0.112**        | 0.126      |

NeuralDTMC produces **sharper distributions** and significantly better numerical accuracy, while traditional DTMC remains slightly better calibrated.



## Key Takeaways

* NeuralDTMC **does not change the DTMC itself**, only how transitions are learned and decoded
* **Argmax decoding** is used for most-likely-state metrics (Top-k)
* **Expectation decoding** is used for numerical error metrics (MAE / RMSE)
* This separation makes evaluation **more principled and interpretable**


