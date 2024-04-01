#### Regression analysis ----

# install.packages("devtools")
# devtools::install_github("JustinMShea/wooldridge")

install.packages("woodlridge")
library(wooldridge)

install.packages("tidyverse")
library(tidyverse)

data(wage1)

wage1 %>% head()

?wage1

# Estimate wage using years of education


model <- lm(formula = wage ~ educ, data = wage1)
model

# Interpretation


# Predict average hourly salary if education is 16 years
wage_16 <-  -0.9049 + 0.5414 * 16
print(wage_16)

# Gauss-Markov Theorem 
# The OLS estimates are BLUE: Best Linear Unbiased Estimators

summary(model)

# If we are dealing a 95% confidence level, then it's significance level is 5% or 0.05,
# Compare 0.187 to 0.05. If it is less than, then we reject null hypothesis
# Null hypothesis is estimate is not significant, so when we reject it it's not significant
# Here we don't reject null hypothesis and therefore the intercept is not significant in the model
# Education is less that  2 x 10 ^ -16, means it's less than 0.05 and we can reject null hypothesis. 
# Regression coefficient for education is therefore significant

# Multiple R-Squared
0.1648 * 100

# It only means that the regression equation fits the data by 16.48%
# Not a really good fit
# Any variation in education only explains 16.48% variation in the average hourly earnings

# Confidence Interval
?confint
confint(model, level = 0.95)

# Do we find the intercept (-0.9049) between -2.25 and 0.44? yes!


# Multiple Linear Regression
model1 <- lm(formula = wage ~ educ + exper, data = wage1)
model1

# Let's predict wage if education is 18 years and experience is 4 years
wage_18ed_4work <- -3.3905 + 0.6443 * 18 + 0.0701 * 4

summary(model1)

# Multiple R-squared:
# The equation with educ and experience fits the model by 22.52%
# The closeness of R-Squared and Adjusted R-Square, 
# Adjusted R-Squared means

plot(model1)


#### visualizing regression models in R ----

install.packages("performance")
library(performance)

install.packages("effects")
library(effects)

install.packages("estimability")
library(estimability)

install.packages("devtools")
devtools::install_github("strengejacke/sjPlot")

install.packages("sjPlot")
library(sjPlot)

install.packages("ISLR")
library(HistData)

install.packages("HistData")
library(ISLR)

data(mtcars)
data(wage1) # from wooldridge
data(Wage) # from ISLR

data(GaltonFamilies)

model_galton <- lm(formula = childHeight ~ midparentHeight, data = GaltonFamilies)
summary(model_galton)
plot(model_galton)

model_wage1 <- lm(formula = wage ~ educ, data = wage1)
summary(model_wage1)
plot(model_wage1)


model_mpg <- lm(formula = mpg ~ cyl + drat, data = mtcars)
plot(model_mpg)
summary(model_mpg)

# Performance package:
performance::check_model(model_wage1)

# Effects package:
plot(allEffects(model_wage1), grid = True)

predictor()

# sjPlot package:
sjPlot::plot_model(model_wage1)



# unemployment is available
summary(unemployment)

# newrates is available
newrates

# Predict female unemployment in the unemployment dataset
unemployment$prediction <-  predict(unemployment_model)

# Load the ggplot2 package
library(ggplot2)

# Make a plot to compare predictions to actual (prediction on x axis)
ggplot(unemployment, aes(x = prediction, y = female_unemployment)) + 
  geom_point() +
  geom_abline(color = "blue")

# Predict female unemployment rate when male unemployment is 5%
pred <- predict(unemployment_model, newdata = newrates)
pred


# bloodpressure is available
summary(bloodpressure)

# bloodpressure_model is available
bloodpressure_model

# Predict blood pressure using bloodpressure_model: prediction
bloodpressure$prediction <- predict(bloodpressure_model)


library(ggplot2)

# Plot the results
ggplot(GaltonFamilies, aes(x = midparentHeight, y = childHeight)) + 
  geom_point() +
  geom_abline(color = "blue")
