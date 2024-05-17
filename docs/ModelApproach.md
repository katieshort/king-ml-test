# Approach Summary: Doubly Robust Learning for Game Design Recommendation

## Objective

The goal is to recommend a game design (A or B) for each player that maximizes future monetization (`n14`) and without
any decline in
engagement intensity (`n13`).

## Challenge: Counterfactual Predictions

The main challenge arises from the need to make counterfactual predictions, to estimate potential outcomes if a player
exposed to one game design had instead experienced the other.
This is complicated by the nature of the dataset, particularly the non-random assignment of players to game
designs A and B, and the almost perfect separation in pre-test features.

### Non-Randomized Group Assignment

The dataset seems to exhibit the critical issue of "perfect separation", where players can be perfectly
classified into groups based on pre-test features. There is a clear cut-off in the pre-test engagement metrics `n1`, as
well as other numerical features. For categorical features, there is also minimal overlap with only one common category
between groups for all features.

### Implications of Perfect Separation

Perfect separation leads to several problems in standard causal inference methods:

- **Propensity Score Estimation**: The sharp distinction between groups means that estimating propensity scores, i.e.
  the probability of receiving a particular game design given the
  covariates, becomes problematic. In extreme cases, the propensity score model can output probabilities close to 0 or
  1, reducing the effectiveness of methods like weighting or matching that rely on these scores.
- **Model Validity and Extrapolation in Outcome Modelling**:  When there is little overlap in the covariate
  distributions between the groups, it challenges
  the assumptions necessary for causal models to provide unbiased estimates of treatment effects. This lack of
  overlap (common support) complicates the outcome modelling:
    - **Extrapolation Risks**: Predicting counterfactual outcomes requires the model to extrapolate beyond its observed
      combinations of covariates and
      treatments. This is because the model, although it may have observed all ranges of
      values for features under treatment 'A', has never seen these same feature ranges under treatment 'B'.
      This scenario forces the model to make predictions in parts of the feature space where it has no data, increasing
      the risk of biased or unreliable predictions.
    - **Potential for Reliable Interpolation**: Near the boundaries where treatment assignments change, the model may
      more accurately interpolate. Here, players with covariate values close to the threshold for switching treatments
      could have more reliable counterfactual predictions, as extrapolation from the observed data is minimal.

## Model Validation Challenges

Validating causal models in this scenario is particularly tricky due to the lack of ground truth for
counterfactual outcomes. Since we can't observe both potential outcomes for any single player, assessing model accuracy
involves:

- **Cross-Validation**: to assess the predictive
  accuracy of the outcome models on the known outcomes.
- **Overlap in Propensity Scores**: checking for overlap in propensity scores between the treatment groups to ensure
  there is sufficient common support.
- **Sensitivity Analysis**: Conducting sensitivity analyses to explore how changes in the model's assumptions might
  affect the estimates. This can help identify whether the conclusions are robust to different configurations of the
  model or data.

## Modelling Approach

### Initial experiments

I explored several approaches, summarised briefly below:

- **T-learner**:
    - This approach trains two separate outcome models for each treatment group (Design A and Design B). Counterfactual
      predictions are made by applying the model trained on
      one group to predict outcomes for the other group.
    - **Limitations**: The perfect separation in the dataset means each model only learns from its respective group's
      data.
      Predicting counterfactuals thus requires extrapolation from one
      group
      to another, risking biases due to the models never observing covariate combinations across different
      treatments.
- **Double Machine Learning**:
    - DML uses separate machine learning models to predict the treatment and the outcome
      based on covariates, then estimates the treatment effect from the residuals of these predictions. Counterfactuals
      are estimated by adjusting the observed outcomes using these estimated effects.
    - **Limitations**:  With perfect
      separation in our dataset, the treatment prediction model within DML can predict treatment
      assignment with zero error, leading to zero residuals for the treatment model. This undermines the residual-based
      analysis crucial to DML, making it impossible to regress outcome residuals on treatment residuals to estimate the
      causal effect.

### Final Model: Doubly Robust Learning

Given the limitations faced with T-learner and DML, I chose the Doubly Robust Learning approach, combining the strengths
of propensity score weighting with outcome modeling. This method is 'doubly robust' in that it can provide reliable
causal estimates if either the model for the propensity score or the outcome is correctly specified.

#### Methodology

1. **Propensity Score Estimation**: Calculate the propensity scores for each player to be assigned to Design B based on
   observed covariates,
   balancing the groups and ensuring fair comparisons between designs.
2. **Outcome Modeling**: Model the potential outcomes for both monetization and engagement for each design,
   using players’ baseline characteristics and their calculated propensity scores. This step involves fitting a
   regression model for each outcome, adjusted by the propensity scores to account for the baseline covariates.
3. **Counterfactual Prediction**: For each player, we predict the potential outcomes under both game designs using the
   fitted models. This step allows us to estimate what each player’s engagement and monetization would likely be if they
   were assigned to either design.
4. **Game Design Recommendation**: For each player, recommend the design that predicts the highest
   future monetization (`n14`), provided
   there is no predicted decline in engagement (`n13`).

### Model Selection

In terms of the outcome models, the choice between tree-based and linear models presented a trade-off:

- **Tree-based models** excel at capturing complex, nonlinear interactions within the data but typically do
  not extrapolate well outside the training set range. This limits their ability to predict
  outcomes in new, unseen scenarios.
- **Linear models** offer better extrapolation capabilities and can predict beyond the observed data ranges, making them
  useful for scenarios not well-represented in the training data. However, they may fail to capture more
  complex relationships as effectively as tree-based models, and showed poorer performance in cross-validation of the
  outcome model.

For the propensity score model, a logistic regression model was chosen for its ability to provide moderate and stable..
Unlike more complex models, logistic regression did not yield extreme propensity scores due to
overfitting, thus maintaining some degree of overlap in scores between treatment groups.

### Discussion on Limitations

Although the DR method is robust to some extent of misspecification in the propensity or outcome models, the strong
separation still presents a unique challenge. The technique relies on sufficient overlap in propensity scores between
treated and control groups to correctly estimate causal effects.
Moreover, the outcome model must often extrapolate
beyond the observed data to predict counterfactuals, which can increase the risk of biased estimates, particularly when
covariate combinations are not uniformly represented across treatment groups.

## Conclusions and Future Directions

While Doubly Robust Learning provides a sophisticated framework for causal inference, its application to this
dataset—with inherent challenges such as perfect separation—necessitates cautious interpretation, and should be
accompanied by rigorous sensitivity analysis and validation to confirm the robustness of the findings.

Future improvements could include:

- **Enhanced Validation**: Implementing sensitivity analyses, refutation tests, and simulation of treatment effects to
  better validate the model’s predictions.
- **Alternative Methods**: Exploring domain adaptation techniques to better handle
  differences in distributions between the two treatment groups. This method involves adjusting the training process to
  account for the disparate distributions in the treatment and control groups, typically by reweighting or transforming
  the data to reduce bias and improve causal inference accuracy
