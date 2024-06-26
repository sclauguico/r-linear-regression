##### PRELIMINARY DATA ANALYSIS ----
# For restarting environment
rm(list = ls())

PWD <- file.path(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd(PWD)

##### Load packages ----
# wage1 dataset can be obtained from this package
install.packages("wooldridge")
library(wooldridge)

##### Import data and store to df_wage ----
data(wage1)

# Explore more about the dataset
?wage1

# Store GaltonFamilies dataset to df_wage
df_wage <- wage1

# View dataset
View(df_wage)

# View the statistical summary of the height dataset
summary(df_wage)

# Get the number of dimensions or dataset size
dim(df_wage)

##### EDA ----

par(mfrow = c(2,2))
plot(df_wage$educ, df_wage$wage)
plot(df_wage$exper, df_wage$wage)
plot(df_wage$tenure, df_wage$wage)

# INTERPRETATION:
# Seems like educ is potentially a better predictor for predicting the wage

##### MODEL TRAINING ----
#### Split dataset to train/test data ----
# Get the number of rows by using nrow to and store to N
N <- nrow(df_wage)
# Show N
N

# Get 70% of N to split the dataset into 70% for training and the remaining 30% for testing
# Use round() to get an integer
# Enclosing the whole variable declaration will print the value of the variable
(smp_size <- floor(0.70 * N))

# Set the seed to make your partition reproducible
set.seed(143)

train_indeces <- sample(seq_len(N), size = smp_size)

# Split dataset
df_train <- df_wage[train_indeces, ]
df_test <- df_wage[-train_indeces, ]

# Use nrow() to examine df_wage_train_train and df_wage_train_test
nrow(df_train)
nrow(df_test)

#### Train a model using the train data ----
# Show df_wage_train stats summary
summary(df_train)

# Create a formula to express wage as a function of educ: fmla and print it.
(fmla <- wage ~ educ)

# Use lm() to build a model height_model from df_wage_train that predicts the child's height from the parent's mid height 
height_model <- lm(fmla, data = df_train)

##### VISUALIZE THE MODEL ----
# Step 4: Diagnostics
# Assumptions of Linear Regression and Diagnosing Violations of Those Assumptions

#  Why?
# We need to check these assumptions when performing linear regression
# as validations of these assumptions can have substantial impact on the parameter
# estimates, including levels of stats sig

# What?
# 1. L - Linear: There is a linear relationship between x and y. If not linear model won't be a good fit for the data and the parameter estimates are meaningless
# 2. I - Independent variables: X1 and X2 should not be perfect collinear. This tells if feature selection is needed. Often violated when the data are collected over time , like time-series data, as successive residuals tend to be positively correlated like temp in a city, which is correlated with the season and unlikely to be random
# 3. N - Normal: Residuals are normally distributed, nice-to-have but not required
# 4. E - Equal Variance: Homoscedasticity. Error term (residuals is used to estimate error term) is normally dist. If assumption is violated regression model maybe a poor fit

# How?
# Assumptions 1 and 4
# Residuals vs Fitted:
# A residual plot is a plot of the residuals on the y-axis vs the predicted values of the dep var on the x-axis
# Points should be symmetrically distributed around the line, ideally spread out without pattern
# If residuals met the assumption that they are normally distributed with mean zero, 
# then the trend line should closely follow the y equals zero line on the plot.
# Red line is LOESS trend line, smooth curve following the data

# If a linear regression model is a good fit, 
# then the residuals are approximately normally distributed, with mean zero.
# This means that the error is not random.

# If there is a curve linear trend on the residuals, it suggests that the relationship between x and y is not linear
# and we should not use a straight line to model this data

# Assumption 2:
# Can be observed by using Pearson correlation as part of EDA, and handled by Feature Selection techniques in Feature Engineering

# Assumption 3
# Q-Q Plot:
# It shows whether or not the residuals follow a normal distribution.
# On the x-axis: The points are quantiles of the normal dist having the same mean and variance. 
# On the y-axis: The standardized residuals, which are the residuals divided by their standard deviation.
# If the data follows a normal distribution,  the points will fall close to a straight diagonal line. 
# If the data deviates from normality, the points will fall away from the line.

# If there is a bow-shape in the data points compared to the line, the deviations from the diagonal
# line indicates that the residuals have excessive skewness and are not symmetrically distributed
# and have many log errors on the positive direction if data points are above the line, negative below
# S-shaped - shows a distributions that is under dispersed first relative to normal distribution
# or like a uniform distribution
# Bow- and S- shaped dists indicated that the regression residuals are not normally distributed, which violates assumption 3

# Assumptions 1 and 4
# Scale-Location
# It shows the square root of the standardized residuals versus the fitted values. 
# Can be symmetrical, but if funneled, then variance is not equal

# Residuals vs Leverage
#  Tells which observations are influential. Resuduals with high leverage can skew results

par(mfrow = c(2,2))
plot(height_model)

# INTERPRETATION:
# Step 7: Interpret regression models
# 1. The plot in the image shows that the red line approximately follows the zero line plot.
# The residuals appear to be randomly scattered around the horizontal line at zero, which is a good sign.
# They are symmetrically distributed which means that the residuals met the assumption that they are normally distributed with approx. mean equal to 0.
# This suggests that the model meets the assumptions of homoscedasticity (constant variance) and independence of errors. 
# Equal variance means the prediction, closely follows the linear model closely

# 2. Most of the data points follow the line closely. Three points at each extreme don't follow the line, namely:
# point 817, 128, and 293, which correspond to the row of the dataset where the bad residuals occur.
# The linear portion in the middle suggests the data have more extreme values (or outliers) than the normal distribution in the tails
# In the left and right most side of the plot, the residuals are larger than expected. Poor fit for the extremely taller and smaller
# midparentHeight

# 3. The scale-location plot. Residuals are scattered. This suggests that the model meets the assumptions of homoscedasticity  

# 4. The residuals vs leverage plot in the image doesn't show any clear pattern, which is a good sign. 
# There are no outliers with high leverage that are heavily influencing the model.
# Sample 230 has leverage and label 128 and 859 has large residual, label 859 has high leverage, which can skew the results

# install.packages("performance")
library(performance)

# Performance package:
performance::check_model(height_model)


##### INTERPRET THE MODEL ----

# INTIRPRETATION:
# 1. it looks like the model-predicted data (green line) matches the observed data (blue line) reasonably well, 
# except at the center of the plot.

# 2. The residuals appear to be scattered. This suggests that the
# residuals are independent of the fitted values. 

# Use summary() to examine the model
summary(height_model)

# INTERPRETATION:


#### Evaluate model using Holdout Validation ----

# Manual prediction
wage_pred <- -1.0683 + 0.56140 * 16

# Evaluate model using Holdout Validation ----
df_train$pred <- predict(height_model, newdata = df_train)
df_test$pred <- predict(height_model, newdata = df_test)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
mae_train <- mean(abs(df_train$pred - df_train$wage))
mae_test <- mean(abs(df_test$pred - df_test$wage))
mse_train <- mean((df_train$pred - df_train$wage)^2)
mse_test <- mean((df_test$pred - df_test$wage)^2)
rmse_train <- RMSE(df_train$pred, df_train$wage)
rmse_test <- RMSE(df_test$pred, df_test$wage)
rsq_train <- cor(df_train$pred, df_train$wage)^2
rsq_test <- cor(df_test$pred, df_test$wage)^2

# Plot the predictions (on the x-axis) against the outcome (wage) on the test data
# Visualize predictions vs. actuals
ggplot(df_test, aes(x = pred, y = wage)) + 
  geom_point() + 
  geom_abline() +
  geom_smooth(method = 'lm') +
  labs(title = "Predicted vs Actual Wage (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
set.seed(143)
cv_results <- train(wage ~ educ, data = df_wage, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_mae <- cv_results$results$MAE
cv_mse <- cv_results$results$MSE
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("MAE (Train):", mae_train, "\n")
cat("MAE (Test):", mae_test, "\n")

cat("MSE (Train):", mse_train, "\n")
cat("MSE (Test):", mse_test, "\n\n")

cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")

cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
cat("MAE (CV):", mean(cv_mae), "\n")
cat("MSE (CV):", mean(cv_mse), "\n")
cat("RMSE (CV):", mean(cv_rmse), "\n")
cat("R-squared (CV):", mean(cv_rsq), "\n")

# INTERPRETATION:
# Holdout Validation:

# RMSE (Root Mean Squared Error):
#   Train Set: RMSE is 3.47, indicating, on average, the model's predictions on the training set are off by approximately 3.47 units of the target variable.
#   Test Set: RMSE is 3.19, showing that, on average, the model's predictions on the unseen test data are off by approximately 3.19 units of the target variable.
# R-squared:
#   Train Set: R-squared is 0.10, indicating that approximately 10% of the variance in the target variable is explained by the model on the training data.
#   Test Set: R-squared is 0.11, suggesting that approximately 11% of the variance in the target variable is explained by the model on the test data.


# Cross-Validation:

#   RMSE (Root Mean Squared Error):
#     Cross-Validation RMSE is 3.39, which is the average RMSE across all folds of the cross-validation process. It represents the model's generalization error.
#   R-squared:
#     Cross-Validation R-squared is 0.10, indicating the proportion of variance in the target variable that the model explains on average across all folds of the cross-validation process.
# Overall, the model's performance seems consistent between the Holdout Validation and Cross-Validation results. However, the model's performance, as indicated by the R-squared values, is relatively low, suggesting that the model explains only a small portion of the variance in the target variable. This might imply that the model may not capture all relevant features or that the relationship between features and the target variable is inherently complex. Further model refinement or feature engineering may be necessary to improve performance.

##### MULTILINEAR REGRESSION ----

# Create a formula to express wage as a function of educ: fmla and print it.
# Create a formula to express wage as a function of educ: fmla and print it.
(fmla <- wage ~ educ + exper)

# Use lm() to build a model height_model from df_wage_train that predicts the child's height from the parent's mid height 
height_model <- lm(fmla, data = df_train)


##### VISUALIZE THE MODEL ----
par(mfrow = c(2,2))
plot(height_model)

# INTERPRETATION:
# 1. the residuals appear to be scattered somewhat randomly around the zero line, but there might be a slight curve. 
# This suggests that the model might not be perfect, but it could be a reasonable fit for the data.

# 2. the points deviate from the line somewhat, which suggests that the errors may not be perfectly normal.

# 3. There doesn't seem to be a clear pattern between the residuals and the leverage scores, suggesting that there aren't 
# any outliers with high leverage that are exerting undue influence on the model.

# 4. There doesn't seem to be a clear pattern between the residuals and the leverage scores, suggesting that
# there aren't any outliers with high leverage that are exerting undue influence on the model.

library(performance)

# Performance package:
performance::check_model(height_model)

# INTERPRETATION:
# 1. 

# 2.

# 3. 

# 4. 

# 5.

# 6.

##### INTERPRET THE MODEL ----

# Use summary() to examine the model
summary(height_model)

# INTERPRETATION:


#### Evaluate model using Holdout Validation ----

# Manual prediction
wage_pred <- 18.09189 + 0.70359 * 69.020

# Evaluate model using Holdout Validation ----
df_train$pred <- predict(height_model, newdata = df_train)
df_test$pred <- predict(height_model, newdata = df_test)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
mae_train <- mean(abs(df_train$pred - df_train$wage))
mae_test <- mean(abs(df_test$pred - df_test$wage))
mse_train <- mean((df_train$pred - df_train$wage)^2)
mse_test <- mean((df_test$pred - df_test$wage)^2)
rmse_train <- RMSE(df_train$pred, df_train$wage)
rmse_test <- RMSE(df_test$pred, df_test$wage)
rsq_train <- cor(df_train$pred, df_train$wage)^2
rsq_test <- cor(df_test$pred, df_test$wage)^2

# Plot the predictions (on the x-axis) against the outcome (wage) on the test data
# Visualize predictions vs. actuals
ggplot(df_test, aes(x = pred, y = wage)) + 
  geom_point() + 
  geom_abline() +
  geom_smooth(method = 'lm') +
  labs(title = "Predicted vs Actual Wage (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
set.seed(143)
cv_results <- train(wage ~ educ + exper, data = df_wage, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_mae <- cv_results$results$MAE
cv_mse <- cv_results$results$MSE
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("MAE (Train):", mae_train, "\n")
cat("MAE (Test):", mae_test, "\n")

cat("MSE (Train):", mse_train, "\n")
cat("MSE (Test):", mse_test, "\n\n")

cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")

cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
cat("MAE (CV):", mean(cv_mae), "\n")
cat("MSE (CV):", mean(cv_mse), "\n")
cat("RMSE (CV):", mean(cv_rmse), "\n")
cat("R-squared (CV):", mean(cv_rsq), "\n")

# INTERPRETATION:
# When the R-squared is higher in holdout validation than in cross-validation, 
# it could be a sign of overfitting, and further steps such as regularization or 
# simplifying the model may be necessary to improve generalization performance.
