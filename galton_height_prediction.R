##### PRELIMINARY DATA ANALYSIS ----
# For crestarting environment
rm(list = ls())

##### Load packages ----
# GaltonFamilies dataset can be obtained from this package
# install.packages("HistData")
library(HistData)

##### Import data and store to df_height ----
data(GaltonFamilies)

# Explore more about the dataset
?GaltonFamilies

# Store GaltonFamilies dataset to df_height
df_height <- GaltonFamilies

# View dataset
View(df_height)

# View the statistical summary of the height dataset
summary(df_height)

# Get the number of dimensions or dataset size
dim(df_height)

##### EDA ----

par(mfrow = c(2,2))
plot(df_height$father, df_height$childHeight)
plot(df_height$mother, df_height$childHeight)
plot(df_height$midparentHeight, df_height$childHeight)

# INTERPRETATION:
# Seems like midparentHeight is potentially a better predictor for predicting the childHeight

##### FEATURE ENGINEERING ----
# Understand the Feature Engineered variable
# mideparentHeight = (father + 1.08*mother)/2
m0 <- lm(midparentHeight ~ father + mother, data = df_height)
summary(m0)

# Copy dataset for EDA purposes
df_height_eda <- df_height

# Create a new column pred for prediction
df_height_eda$pred <- predict(m0, newdata = df_height_eda)

# Plot the predictions (on the x-axis) against the outcome (midparentHeight)
ggplot(df_height_eda, aes(x = pred, y = midparentHeight)) + 
  geom_point() + 
  geom_abline()
  
# INTERPRETATION:
# We can predict midparentHeight witt this equation:
# midparentHeight = 0.5 (father) + 0.54 (mother)
# PERFECT FIT
par(mfrow = c(2,2))
plot(m0) 

# INTERPRETATION
# Normal Q-Q: In a normal Q-Q plot, if the data follows a normal distribution, 
# the points will fall close to a straight diagonal line. If the data deviates from normality, 
# the points will fall away from the line.

# The points deviate from the straight diagonal line, which suggests that the data is not normally distributed. 
# Specifically, the points in the tails are farther away from the line, which suggests that the data may have 
# heavier tails than a normal distribution. This means that there are more extreme values in the data than would 
# be expected in a normal distribution.
# These are called outliers can be addressed using outlier-handling techniques

# The red line shows a slight upward trend as the fitted values increase. This suggests that the variance of the residuals might be increasing with the fitted values, violating the assumption of homoscedasticity.
# There is also some curvature in the red line, which could indicate a non-normal distribution of errors.
# Overall: The scale-location plot in the image suggests that the linear regression model might not meet the assumptions of homoscedasticity and normality of errors. 

# There are no data points far from the center on the x-axis, so leverage doesn't seem to be a major concern. 
# However, it's difficult to say definitively without the actual data or Cook's distance values.

##### MODEL TRAINING ----
#### Split dataset to train/test data ----
# Get the number of rows by using nrow to and store to N
N <- nrow(df_height)
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
df_train <- df_height[train_indeces, ]
df_test <- df_height[-train_indeces, ]

# Use nrow() to examine df_height_train_train and df_height_train_test
nrow(df_train)
nrow(df_test)

#### Train a model using the train data ----
# Show df_height_train stats summary
summary(df_train)

# Create a formula to express childHeight as a function of midparentHeight: fmla and print it.
(fmla <- childHeight ~ midparentHeight)

# Use lm() to build a model height_model from df_height_train that predicts the child's height from the parent's mid height 
height_model <- lm(fmla, data = df_train)

# Use summary() to examine the model
summary(height_model)

# INTERPRETATION:


#### Evaluate model using Holdout Validation ----

# Manual prediction
childHeight_pred <- 18.09189 + 0.70359 * 69.020

# Evaluate model using Holdout Validation ----
df_train$pred <- predict(height_model, newdata = df_train)
df_test$pred <- predict(height_model, newdata = df_test)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
rmse_train <- RMSE(df_train$pred, df_train$childHeight)
rmse_test <- RMSE(df_test$pred, df_test$childHeight)
rsq_train <- cor(df_train$pred, df_train$childHeight)^2
rsq_test <- cor(df_test$pred, df_test$childHeight)^2

# Plot the predictions (on the x-axis) against the outcome (childHeight) on the test data
# Visualize predictions vs. actuals
ggplot(df_test, aes(x = pred, y = childHeight)) + 
  geom_point() + 
  geom_abline() +
  labs(title = "Predicted vs Actual Child Height (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
cv_results <- train(childHeight ~ midparentHeight, data = df_height, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")
cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
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



#### Visualize the linear regression model
par(mfrow = c(2,2))
plot(height_model)

# INTERPRETATION:
# 1. The plot in the image shows a slight curvature, which suggests that the linear model might not be the best fit 
# for the data. A curvilinear trend might be better captured with a polynomial regression model.

# 2. The Q-Q plot in the image deviates from a straight line, 
# which suggests that the residuals may not be normally distributed.

# 3. The scale-location plot in the image shows a slight funnel shape, which suggests that the 
# variance of the errors might be increasing with the fitted values. 
# This is a violation of the homoscedasticity assumption. 

# 4. The residuals vs leverage plot in the image doesn't show any clear pattern, which is a good sign. 
# There are no outliers with high leverage that are heavily influencing the model.

# install.packages("performance")
library(performance)

# Performance package:
performance::check_model(height_model)

# INTERPRETATION:



#### Other models ----

# Create a formula to express childHeight as a function of midparentHeight: fmla and print it.
# Create a formula to express childHeight as a function of midparentHeight: fmla and print it.
(fmla <- childHeight ~ midparentHeight + gender)

# Use lm() to build a model height_model from df_height_train that predicts the child's height from the parent's mid height 
height_model <- lm(fmla, data = df_train)

# Use summary() to examine the model
summary(height_model)

# INTERPRETATION:


#### Evaluate model using Holdout Validation ----

# Manual prediction
childHeight_pred <- 18.09189 + 0.70359 * 69.020

# Evaluate model using Holdout Validation ----
df_train$pred <- predict(height_model, newdata = df_train)
df_test$pred <- predict(height_model, newdata = df_test)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
rmse_train <- RMSE(df_train$pred, df_train$childHeight)
rmse_test <- RMSE(df_test$pred, df_test$childHeight)
rsq_train <- cor(df_train$pred, df_train$childHeight)^2
rsq_test <- cor(df_test$pred, df_test$childHeight)^2

# Plot the predictions (on the x-axis) against the outcome (childHeight) on the test data
# Visualize predictions vs. actuals
ggplot(df_test, aes(x = pred, y = childHeight)) + 
  geom_point() + 
  geom_abline() +
  labs(title = "Predicted vs Actual Child Height (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
cv_results <- train(childHeight ~ midparentHeight, data = df_height, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")
cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
cat("RMSE (CV):", mean(cv_rmse), "\n")
cat("R-squared (CV):", mean(cv_rsq), "\n")

# INTERPRETATION:
# When the R-squared is higher in holdout validation than in cross-validation, 
# it could be a sign of overfitting, and further steps such as regularization or 
# simplifying the model may be necessary to improve generalization performance.



##### IMPROVING THE MODELS
# Option 1: Transform childHeight (logarithmic)
df_height_transformed <- df_height
df_height_transformed$childHeight_log <- log(df_height_transformed$childHeight)

df_train_transformed <- df_height_transformed[train_indeces, ]
df_test_transformed <- df_height_transformed[-train_indeces, ]

# Train model on transformed data
height_model_transformed <- lm(childHeight_log ~ midparentHeight, data = df_train_transformed)

# Evaluate model using Holdout Validation ----
df_train_transformed$pred <- predict(height_model_transformed, newdata = df_train_transformed)
df_test_transformed$pred <- predict(height_model_transformed, newdata = df_test_transformed)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
rmse_train <- RMSE(df_train_transformed$pred, df_train_transformed$childHeight)
rmse_test <- RMSE(df_test_transformed$pred, df_test_transformed$childHeight)
rsq_train <- cor(df_train_transformed$pred, df_train_transformed$childHeight)^2
rsq_test <- cor(df_test_transformed$pred, df_test_transformed$childHeight)^2

# Plot the predictions (on the x-axis) against the outcome (childHeight) on the test data
# Visualize predictions vs. actuals
ggplot(df_test_transformed, aes(x = pred, y = childHeight)) + 
  geom_point() + 
  geom_abline() +
  labs(title = "Predicted vs Actual Child Height (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
cv_results <- train(childHeight ~ midparentHeight, data = df_height_transformed, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")
cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
cat("RMSE (CV):", mean(cv_rmse), "\n")
cat("R-squared (CV):", mean(cv_rsq), "\n")






# Evaluate and compare with original model

# Option 2: Weighted least squares
# install.packages("car")
library(car)
weights <- 1 / (df_train$childHeight^2)  # Example weighting scheme (adjust as needed)
height_model_transformed_weighted <- lm(childHeight ~ midparentHeight, data = df_train, weights = weights)

# Evaluate and compare with original model

# Evaluate model using Holdout Validation ----
df_train$pred <- predict(height_model_transformed_weighted, newdata = df_train)
df_test$pred <- predict(height_model_transformed_weighted, newdata = df_test)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
rmse_train <- RMSE(df_train$pred, df_train$childHeight)
rmse_test <- RMSE(df_test$pred, df_test$childHeight)
rsq_train <- cor(df_train$pred, df_train$childHeight)^2
rsq_test <- cor(df_test$pred, df_test$childHeight)^2

# Plot the predictions (on the x-axis) against the outcome (childHeight) on the test data
# Visualize predictions vs. actuals
ggplot(df_test, aes(x = pred, y = childHeight)) + 
  geom_point() + 
  geom_abline() +
  labs(title = "Predicted vs Actual Child Height (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
cv_results <- train(childHeight ~ midparentHeight, data = df_height, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")
cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
cat("RMSE (CV):", mean(cv_rmse), "\n")
cat("R-squared (CV):", mean(cv_rsq), "\n")

# Option 3: Polynomial regression
height_model_transformed_poly <- lm(childHeight ~ midparentHeight + I(midparentHeight^2), data = df_train)

# Evaluate and compare with original model
# Evaluate model using Holdout Validation ----
df_train$pred <- predict(height_model_transformed_poly, newdata = df_train)
df_test$pred <- predict(height_model_transformed_poly, newdata = df_test)


# Load packages for regression evaluation
# install.packages("caret")
library(caret)

# Evaluation metrics
rmse_train <- RMSE(df_train$pred, df_train$childHeight)
rmse_test <- RMSE(df_test$pred, df_test$childHeight)
rsq_train <- cor(df_train$pred, df_train$childHeight)^2
rsq_test <- cor(df_test$pred, df_test$childHeight)^2

# Plot the predictions (on the x-axis) against the outcome (childHeight) on the test data
# Visualize predictions vs. actuals
ggplot(df_test, aes(x = pred, y = childHeight)) + 
  geom_point() + 
  geom_abline() +
  labs(title = "Predicted vs Actual Child Height (Test Set)")

# INTERPRETATION:

# Cross-Validation Evaluation ----
# Using 5-fold Cross-Validation
cv_results <- train(childHeight ~ midparentHeight, data = df_height, method = "lm", trControl = trainControl(method = "cv", number = 5))
cv_rmse <- cv_results$results$RMSE
cv_rsq <- cv_results$results$Rsquared

# Summary
cat("Holdout Validation:\n")
cat("RMSE (Train):", rmse_train, "\n")
cat("RMSE (Test):", rmse_test, "\n")
cat("R-squared (Train):", rsq_train, "\n")
cat("R-squared (Test):", rsq_test, "\n\n")

cat("Cross-Validation:\n")
cat("RMSE (CV):", mean(cv_rmse), "\n")
cat("R-squared (CV):", mean(cv_rsq), "\n")
