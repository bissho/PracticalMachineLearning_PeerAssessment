Prediction Assignment
========================================================

The goal of the project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. We should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

# Data analisis

## Data load

First I downloaded from 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv (2014/06/20 18:04) and loaded in R:

```r
trainSet = read.csv("data/pml-training.csv")
```


## Selection of predictors

I did a preliminary study in order to reduce the number of predictor(my machine is a Pentium M with Lubuntu (Light Ubuntu)):

-1. First I reduced to columns without lots of NAs or invalid values using summary function. 


```r
dim(trainSet)
names(trainSet)
str(trainSet)
summary(trainSet)
inUse = c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", "gyros_belt_x", 
    "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", 
    "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", 
    "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", 
    "gyros_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", 
    "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", "gyros_dumbbell_x", 
    "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y", 
    "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", 
    "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", "gyros_forearm_x", 
    "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", 
    "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")
trainSet = trainSet[, c(inUse, "classe")]
```


-2. Then I created a matriz of correlation between columns except themselves and outcome in order to keep only one column from groups of correlated ones. 


```r
M <- abs(cor(trainSet[, -51]))
diag(M) <- 0
which(M > 0.7, arr.ind = T)
```

```
##                      row col
## yaw_belt               3   1
## total_accel_belt       4   1
## accel_belt_y           9   1
## accel_belt_z          10   1
## accel_belt_z.1        11   1
## accel_belt_x           8   2
## magnet_belt_x         12   2
## roll_belt              1   3
## total_accel_belt       4   3
## accel_belt_x           8   3
## accel_belt_z          10   3
## accel_belt_z.1        11   3
## magnet_belt_x         12   3
## roll_belt              1   4
## yaw_belt               3   4
## accel_belt_y           9   4
## accel_belt_z          10   4
## accel_belt_z.1        11   4
## magnet_dumbbell_x     35   5
## magnet_dumbbell_y     36   5
## pitch_belt             2   8
## yaw_belt               3   8
## magnet_belt_x         12   8
## roll_belt              1   9
## total_accel_belt       4   9
## accel_belt_z          10   9
## accel_belt_z.1        11   9
## roll_belt              1  10
## yaw_belt               3  10
## total_accel_belt       4  10
## accel_belt_y           9  10
## accel_belt_z.1        11  10
## roll_belt              1  11
## yaw_belt               3  11
## total_accel_belt       4  11
## accel_belt_y           9  11
## accel_belt_z          10  11
## pitch_belt             2  12
## yaw_belt               3  12
## accel_belt_x           8  12
## magnet_belt_z         14  13
## magnet_belt_y         13  14
## gyros_arm_y           20  19
## gyros_arm_x           19  20
## magnet_arm_y          23  22
## magnet_arm_x          22  23
## magnet_arm_z          24  23
## magnet_arm_y          23  24
## accel_dumbbell_y      33  25
## accel_dumbbell_x      32  26
## accel_dumbbell_z      34  27
## accel_dumbbell_y      33  28
## gyros_dumbbell_z      31  29
## gyros_forearm_y       43  29
## gyros_forearm_z       44  29
## gyros_dumbbell_x      29  31
## gyros_forearm_y       43  31
## gyros_forearm_z       44  31
## pitch_dumbbell        26  32
## roll_dumbbell         25  33
## total_accel_dumbbell  28  33
## yaw_dumbbell          27  34
## gyros_belt_x           5  35
## magnet_dumbbell_y     36  35
## gyros_belt_x           5  36
## magnet_dumbbell_x     35  36
## gyros_dumbbell_x      29  43
## gyros_dumbbell_z      31  43
## gyros_forearm_z       44  43
## gyros_dumbbell_x      29  44
## gyros_dumbbell_z      31  44
## gyros_forearm_y       43  44
## magnet_forearm_y      49  46
## accel_forearm_y       46  49
```

```r
reducedTrainSet = trainSet[, -c(3, 4, 8, 9, 10, 11, 12, 14, 20, 23, 24, 32, 
    33, 34, 35, 36, 43, 44, 49)]
```


## Predictor method

First I chose the *tree method* (rpart) as I supossed that it was simple and could be enough with this lot of covariates, but as can be seen it was not enough even inside the training set:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(123232)
modFit <- train(classe ~ ., method = "rpart", data = reducedTrainSet)
```

```
## Loading required package: rpart
```

```r
predictions <- predict(modFit, newdata = trainSet)
confusionMatrix(predictions, trainSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4532  998 1236  549   32
##          B  230  893   64  458   52
##          C  402 1251 1776 1218 1243
##          D  402  655  346  991  649
##          E   14    0    0    0 1631
## 
## Overall Statistics
##                                         
##                Accuracy : 0.501         
##                  95% CI : (0.494, 0.508)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.363         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.812   0.2352   0.5190   0.3081   0.4522
## Specificity             0.800   0.9492   0.7460   0.8749   0.9991
## Pos Pred Value          0.617   0.5262   0.3015   0.3257   0.9915
## Neg Pred Value          0.915   0.8380   0.8801   0.8658   0.8901
## Prevalence              0.284   0.1935   0.1744   0.1639   0.1838
## Detection Rate          0.231   0.0455   0.0905   0.0505   0.0831
## Detection Prevalence    0.374   0.0865   0.3002   0.1551   0.0838
## Balanced Accuracy       0.806   0.5922   0.6325   0.5915   0.7257
```


Then I tried with one that had inside cross validation: *Random forests*. It took more that 2 hours to run in my laptop but as you can see it did perfect inside training set. The random forest method is a combination of tree predictors such that each tree depends on the values of a random vector and independently tested with the same layout for each of these. It is a substantial modification of bagging that builds a long collection of uncorrelated trees and then averages them.


```r
library(caret)
set.seed(123232)
# modFit <- train(classe ~ .,method='rf',data=reducedTrainSet) predictions
# <- predict(modFit,newdata=trainSet)
# confusionMatrix(predictions,trainSet$classe)
```


## Test data prediction

I download the testing data from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv (2014/06/14 18:04) and loaded in R.


```r
# Test predictions
testingSet = read.csv("data/pml-testing.csv")
```


Then prepared the predicted outcome files as specified in the course with a total score of 20/20.


```r
predictionsTest <- predict(modFit, newdata = testingSet)
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
# Only run one time pml_write_files(predictionsTest)
```


## Other thoughts:

I have not tried regression or other linear models because I don't think that they are fitted in this case as the outcome is not continuous and the order of the values A..E of the outcome has no meaning. This is why I have jumped directly to classification trees.
