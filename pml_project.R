library(caret)

# reading the data
# NA strings are NA, #DIV/0 and blank_strings
pml.train <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
pml.test <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

# partitioning the datasets into parts, 70-30
intrain <- createDataPartition(pml.train$classe, p = 0.7, list = FALSE)
training <- pml.train[intrain, ]
testing <- pml.train[-intrain, ]
rm(intrain)

# removing the predictors with near zero variance, as they do not help much
nzv <- nearZeroVar(training)
training <- training[ , -nzv]
testing <- testing[ , -nzv]
rm(nzv)

# removing the Columns which have > 95% missing data
null <- colSums(is.na(training))/nrow(training)
training <- training[ , null < 0.95]
testing <- testing[ , null < 0.95]
rm(null)

# not any useful data is contained in the first five columns
training <- training[ , -(1:5)]
testing <- testing[ , -(1:5)]

# preprocess_model <- preProcess(training, method = c("center", "scale"))
# training <- predict(preprocess_model, training)
# testing <- predict(preprocess_model, testing)
# Pre-Processing is not necessary in this type of data.
# Maybe because it is already preprocessed,
# or maybe because we are not considering the missing data & near zero variant datas


# random-forest model
set.seed(1345677)
tr_ctrl <- trainControl(method = "cv", number = 5)
modrf <- train(classe ~ ., data = training, method = "rf", trControl = tr_ctrl)
test_pred <- predict(modrf, testing)
final_pred <- predict(modrf, pml.test)
rm(tr_ctrl)
# rm(preprocess_model)

# this gives an accuracy of 100%
act <- c ("B", "A", "B", "A", "A", "E", "D", "B", "A", "A", "B", "C", "B", "A", "E", "E", "A", "B", "B", "B")
test_mat <- confusionMatrix(test_pred, testing$classe)
final_mat <- confusionMatrix(final_pred, act)