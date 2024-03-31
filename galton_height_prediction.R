#### Load packages ----
# GaltonFamilies dataset can be obtained from this package
install.packages("HistData")
library(HistData)

#### Import data and store to df_height ----
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

#### Split dataset to train/test data ----
# Get the number of rows by using nrow to and store to N
N <- nrow(df_height)
# Show N
N

# Get 70% of N to split the dataset into 70% for training and the remaining 30% for testing
# Use round() to get an integer
# Enclosing the whole variable declaration will print the value of the variable
(target <- round(N * 0.70))

# Create the vector of N uniform random variables: gp
gp <- runif(N)

# Use gp to create the training set: df_height_train (70% of data) and df_height_test (25% of data)
df_height_train <- df_height[gp < 0.70, ]
df_height_test <- df_height[gp >= 0.70, ]

# Use nrow() to examine df_height_train_train and df_height_train_test
nrow(df_height_train)
nrow(df_height_test)

#### Train a model using the train data ----
# Show df_height_train stats summary
summary(df_height_train)

# Create a formula to express childHeight as a function of midparentHeight: fmla and print it.
(fmla <- childHeight ~ midparentHeight)

# Use lm() to build a model height_model from df_height_train that predicts the child's height from the parent's mid height 
height_model <- lm(fmla, data = df_height)

# Use summary() to examine the model
summary(height_model)

# INTERPRETATION:


#### Evaluate model using Holdout Validation ----
# Examine the objects that have been loaded
ls.str()

# Manual prediction
childHeight_pred <- 22.63624 + 0.63736 * 75.430

# Predict childHeight from mideparentHeight for the training set using predict function
df_height_train$pred <- predict(height_model, newdata = df_height_train)

# Predict childHeight from mideparentHeight for the test set using predict function
df_height_test$pred <- predict(height_model, newdata = df_height_test)

# Load packages for regression evaluation
install.packages("caret")
library(caret)

# Evaluate the rmse on both training and test data and print them
(rmse_train <- RMSE(df_height_train$pred, df_height_train$childHeight))
(rmse_test <- RMSE(df_height_test$pred, df_height_test$childHeight))

# Create a function that calculates for r-squared
r_squared <- function(predicted_values, actual_values) {
  1 - sum((actual_values - predicted_values)^2) / sum((actual_values - mean(actual_values))^2)
}

# Evaluate the r-squared on both training and test data.and print them
(rsq_train <- r_squared(df_height_train$pred, df_height_train$childHeight))
(rsq_test <- r_squared(df_height_test$pred, df_height_test$childHeight))

# Plot the predictions (on the x-axis) against the outcome (childHeight) on the test data
ggplot(df_height_test, aes(x = pred, y = childHeight)) + 
  geom_point() + 
  geom_abline()


# INTERPRETATION:




#### Evaluate model using Cross Validation----
# Load the package vtreat
library(vtreat)

summary(df_height)

# Get the number of rows in df_height
nRows <- nrow(df_height)

# Implement the 5-fold cross-fold plan with vtreat
splitPlan <- kWayCrossValidation(nRows, 5, NULL, NULL)

# Examine the split plan
str(splitPlan)

# Use Cross-Fold Validation to evaluate model
summary(df_height)

# splitPlan is available
str(splitPlan)

# Run the 3-fold cross validation plan from splitPlan
k <- 5 # Number of folds
df_height$pred.cv <- 0 
for(i in 1:k) {
  split <- splitPlan[[i]]
  model <- lm(childHeight ~ midparentHeight, data = df_height[split$train, ])
  df_height$pred.cv[split$app] <- predict(model, newdata = df_height[split$app, ])
}

# Predict from a full model
df_height$pred <- predict(lm(midparentHeight ~ childHeight, data = df_height))

# Get the rmse of the full model's predictions
RMSE(df_height$pred, df_height$childHeight)

# Get the rmse of the cross-validation predictions
RMSE(df_height$pred.cv, df_height$childHeight)

# INTERPRETATION


#### Visualize the linear regression model
