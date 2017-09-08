library (caret);
library (rpart);
library (rattle);
library (randomForest);

pml.train <- read.csv ("D:/R/JHU/Practical Machine Learning/Course_Project/pml-training.csv", na.strings = c ("NA", "#DIV/0!", ""));
pml.train <- pml.train[ , (colSums(is.na (pml.train)) == 0)];
pml.test <- read.csv ("D:/R/JHU/Practical Machine Learning/Course_Project/pml-testing.csv", na.strings = c ("NA", "#DIV/0!", ""));
pml.test <- pml.test[ , (colSums(is.na (pml.test)) == 0)];
#all the data has been read, and the NA's have also been taken care of

#now for the pre-processing
num_col <- which(lapply(pml.train, class) %in% "numeric");
pre_process_model <- preProcess (pml.train[ , num_col], method = c ('knnImpute', 'center', 'scale'));
pre_process_train <- predict (pre_process_model, pml.train[ , num_col]);
pre_process_train$classe <- pml.train$classe;
pre_process_test <- predict (pre_process_model, pml.test[ , num_col]);

#removing the near-zero variables
#training_data
nzv <- nearZeroVar (pre_process_train, saveMetrics = T);
pre_process_train <- pre_process_train[ , (nzv$nzv == F)]
#testing_data
nzv <- nearZeroVar (pre_process_test, saveMetrics = T);
pre_process_test <- pre_process_test[ , (nzv$nzv== F)];
rm (nzv);

#creating the training & testing dataset
set.seed (10000007);
inTrain <- createDataPartition (pre_process_train$classe, p = 0.7, list = F)
training <- pre_process_train[inTrain, ];
testing <- pre_process_train[-inTrain, ];

#model fitting using random_foresting
model_rf <- train (classe ~., method = "rf", data = training,
					trControl = trainControl (method = 'cv'), number = 5, allowParallel = T, importance = T);

#prediction on testing_data
res_test <- predict (model_rf, testing);
print (confusionMatrix (res_test, testing$classe));

resample_accuracy <- postResample (res_test, testing$classe)
accuracy <- resample_accuracy[[1]];
out_of_sample_error <- 1 - accuracy;
print (paste ("Accuracy = ", accuracy));
print (paste ("Out of sample error = ", out_of_sample_error));
#rf -- accuracy ~ 99.42%
#out_of_sample_error -- 0.57%

#final prediction
res_final <- predict (model_rf, pml.test);
print (res_final);
#E B B A A E E B B E B B B B E E E B E E
#Levels: A B C D E