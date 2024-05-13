## Approach Summary: Doubly Robust Learning for Game Design Recommendation

### Objective

The goal is to recommend a game design (A or B) for each player that would:

1. Maximize future monetization (`n14`).
2. Avoid any decline in engagement intensity (`n13`).

### Methodology

To achieve these objectives, we implement a Doubly Robust Learning approach, which combines the strengths of propensity
score weighting and outcome modeling to provide more reliable causal estimates:

1. **Propensity Score Estimation**: First, we calculate the propensity scores, which is the probability of each player
   receiving their respective game design (A or B) based on observed covariates. This step helps to balance the groups
   and ensure that the comparison of outcomes between designs is as fair as possible.

2. **Outcome Modeling**: Next, we model the potential outcomes for both monetization and engagement for each design,
   using players’ baseline characteristics and their calculated propensity scores. This step involves fitting a
   regression model for each outcome, adjusted by the propensity scores to account for the baseline covariates.

3. **Counterfactual Prediction**: For each player, we predict the potential outcomes under both game designs using the
   fitted models. This step allows us to estimate what each player’s engagement and monetization would likely be if they
   were assigned to either design.

### Decision Rule

For each player, the recommended game design is the one that predicts the highest future monetization (n14), provided
there is no predicted decline in engagement (n13) compared to the alternative design. Currently, the rule is a simple
direct comparison. However, this could be improved by:

- Thresholds for Engagement Decline: Establishing specific thresholds to quantify acceptable levels of engagement
  decline
  when higher monetization is predicted.
- Confidence Intervals: Using confidence intervals to add a layer of statistical certainty to the predictions, ensuring
  that decisions are robust against potential prediction errors.

### Limitations & Concerns

- **Model Dependence**: The reliability of recommendations depends heavily on the correctness of the model
  specifications and the representativeness of the covariates used.
- **Lack of Common Support**: A critical limitation is the potential lack of common support or strong separation between
  the groups, which could significantly undermine the model's ability to make accurate counterfactual estimates. In this
  context, common support refers to the overlap in the distribution of covariates between treated and control groups.
  When there is strong separation, the characteristics of players in each group are distinctly different. This
  separation poses a problem when the model attempts to predict outcomes for a control player as if they had been
  treated—the features of this player might be very different from any treated unit the model was trained on.
  Consequently, the model might extrapolate beyond the range of data it has learned from, leading to unreliable and
  potentially biased predictions.
- **Sensitivity to Assumptions**: The method's effectiveness is contingent upon the validity of its assumptions about no
  unmeasured confounders and the proper specification of models.

### Model Choice and Future Improvements

#### Model Selection

In this project, the choice between tree-based and linear models presented a trade-off:

- Tree-based models excel at capturing complex, nonlinear interactions and dependencies within the data but typically do
  not extrapolate well to data points outside the range of the training set. This limits their ability to predict
  outcomes in new, unseen scenarios.
- Linear models offer better extrapolation capabilities and can predict beyond the observed data ranges, making them
  useful for scenarios that might not be well-represented in the training data. However, they may fail to capture more
  complex relationships as effectively as tree-based models, and showed poorer performance in cross-fitting of nuisance
  models.

#### Future Improvements

Ideas to improve the model's adaptability and accuracy:

- Hybrid Modeling Approaches: Combine tree-based and linear models to utilize the strengths of both—complex pattern
  recognition and effective extrapolation.
- Domain Adaptation Techniques: Apply domain adaptation to better handle differences in distribution between treatment
  groups. This method involves adjusting the training process to account for the disparate distributions in the
  treatment and control groups, typically by reweighting or transforming the data to reduce bias and improve causal
  inference accuracy.
- Enhanced Validation Techniques: Develop targeted validation strategies that test the model’s extrapolation
  capabilities and adaptability across various domains, including synthetic datasets.

### Conclusion

Doubly Robust Learning provides a sophisticated method for estimating the individual treatment effect for observational
data.
By leveraging both propensity score weighting and outcome modeling, it mitigates biases
due to confounding factors and model misspecification, thereby enhancing the reliability of the causal estimates.
Nevertheless, the application of this method should be accompanied by rigorous sensitivity analysis and validation to
confirm the robustness of the findings.
