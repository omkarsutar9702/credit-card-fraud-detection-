library(gbm ,quietly = TRUE)
library(ranger)
library(caret)
library(caTools)
library(pROC)
library(rpart)
library(rpart.plot)
library(neuralnet)
creditcard <- read.csv("D:/R files/credit card fraud detection/credit_card_fraud/creditcard.csv")

#summary data
dim(creditcard)
names(creditcard)
table(creditcard$Class)
summary(creditcard$Amount)
var(creditcard$Amount)
sd(creditcard$Amount)
sum(is.na(creditcard))

#data manipulation
creditcard$Amount = scale(creditcard$Amount)
newdata <- creditcard[,-c(1)]
head(newdata)

#data modelling
set.seed(80)
data_sample <- sample.split(newdata$Class , SplitRatio = 0.80)
train_data = subset(newdata , data_sample ==TRUE)
test_data = subset(newdata , data_sample==FALSE)
dim(train_data)
dim(test_data)

#fitting logistic regression model
logistic_model <- glm(Class~. , test_data ,family = binomial())
summary(logistic_model)
plot(logistic_model)

#fitting decision tree model
decision_tree <- rpart(Class~. , creditcard , method = "class")
predicted_val <- predict(decision_tree , creditcard , type = "class")
probability <- predict(decision_tree , creditcard , type = "prob")
rpart.plot(decision_tree)

#artificial neural network
ann_model <- neuralnet(Class~. , train_data , linear.output = FALSE)
plot(ann_model)

#gradient boosting model 

#get time to train the GBM MODEL 
system.time(
   model_gbm <- gbm(Class~. , 
                    distribution = "bernoulli"
                    , data= rbind(train_data , test_data)
                    , n.trees = 500
                    , interaction.depth = 3
                    , n.minobsinnode = 100
                    , shrinkage = 0.01
                    , bag.fraction = 0.5
                    , train.fraction = nrow(train_data)/(nrow(train_data)+nrow(test_data))
                    )
)

#determine the best iteration for model 
gbm.iter = gbm.perf(model_gbm  , method = "test")

model_influence <- relative.influence(model_gbm , n.trees = gbm.iter , sort. = TRUE)
plot(model_gbm)

#plot and calculate AUC on test data
gbm_test <- predict(model_gbm ,newdata = test_data,n.trees = gbm.iter )
gbm_auc <- roc(test_data$Class , gbm_test , plot = TRUE , col="red")
print(gbm_auc)














