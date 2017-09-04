Introduction
------------

Personal movement devices such as Jawbone Up, Nike FuelBand, and Fitbit,
are popular and can be used to collect a large amount of data about
personal activity. In this project, we focus on qualifying self movement
to automatically assessing movements in a particular dumbell lifting
simulation. Recognizting particular movements and providing feedback on
how well these movements are executed can significantly assist with
automation of effective training.

In the simulation, six young health participants were asked to perform
one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five
different fashions: exactly according to the specification (Class A),
throwing the elbows to the front (Class B), lifting the dumbbell only
halfway (Class C), lowering the dumbbell only halfway (Class D) and
throwing the hips to the front (Class E). These are also the outcome
variable (designated as 'classe' in the datasets) we want to predict in
this analysis. The data were collected from accelerometers on the belt,
forearm, arm, and dumbell. More information is available from the
website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>

Pre-processing of data
----------------------

Set work directory and download the datasets

    setwd("/Users/zliu11/R_for_DScourses/Course8/CourseProject")
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")

Read datasets, replacing all empty, "NA", and "\#DIV/0!" datapoints with
"NA"

    library(data.table) # load the data.table library
    setwd("/Users/zliu11/R_for_DScourses/Course8/CourseProject")
    trainingDataset <- fread("/Users/zliu11/R_for_DScourses/Course8/CourseProject/pml-training.csv", na.strings = c("NA","",'#DIV/0!'), data.table = F)
    testingDataset <- fread("/Users/zliu11/R_for_DScourses/Course8/CourseProject/pml-testing.csv", na.strings = c("NA","",'#DIV/0!'), data.table = F)

Clean the data, removing all columns (variables) containing "NA".

We have noticed that in the dataset many columns contain NAs. These
columns (variables) don't contribute to the final model. Therefore we
remove these variables.

    trainingDataset <- trainingDataset[,colSums(is.na(trainingDataset)) == 0]
    trainingDataset <- trainingDataset[,8:length(names(trainingDataset))] # only keep relevant variables
    testingDataset <- testingDataset[, colSums(is.na(testingDataset)) == 0]
    testingDataset <- testingDataset[,8:length(names(testingDataset))] # only keep relevant variables

Split the trainingDataset into training and testing subsets.

We further split the trainingDataset into training and testing subsets
for generating the fit models, as well as for conducting cross
validation.

    library(ggplot2); library(caret)
    set.seed(5839)
    inTrain <- createDataPartition(y=trainingDataset$classe, p=0.6, list = F)
    training <- trainingDataset[inTrain,]
    testing <- trainingDataset[-inTrain,]

Model Fitting
-------------

We first try the random forests method, followed by the boosting method.
We will compare the two models and select the one having the better
performance.

### 1. Random Forests

    set.seed(12324)
    library(randomForest)
    modRF <- train(classe ~ ., data = training, method = "rf", trControl=trainControl(method='cv'), number=5)
    modRF$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry, number = 5) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.99%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3344    3    0    0    1 0.001194743
    ## B   17 2250   12    0    0 0.012724879
    ## C    0   24 2024    6    0 0.014605648
    ## D    1    0   43 1885    1 0.023316062
    ## E    0    0    2    7 2156 0.004157044

    predRF <- predict(modRF, testing)
    accuRF <- confusionMatrix(predRF, testing$classe) # calculate the accuracy
    accuRF

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2228    6    0    2    0
    ##          B    4 1503   14    0    0
    ##          C    0    9 1351   24    0
    ##          D    0    0    3 1259    3
    ##          E    0    0    0    1 1439
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9916          
    ##                  95% CI : (0.9893, 0.9935)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9894          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9901   0.9876   0.9790   0.9979
    ## Specificity            0.9986   0.9972   0.9949   0.9991   0.9998
    ## Pos Pred Value         0.9964   0.9882   0.9762   0.9953   0.9993
    ## Neg Pred Value         0.9993   0.9976   0.9974   0.9959   0.9995
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2840   0.1916   0.1722   0.1605   0.1834
    ## Detection Prevalence   0.2850   0.1939   0.1764   0.1612   0.1835
    ## Balanced Accuracy      0.9984   0.9936   0.9912   0.9890   0.9989

### 2. Boosting

    set.seed(2333)
    library(gbm)
    modGBM <- train(classe ~ ., data = training, method ="gbm")
    modGBM$finalModel

    predGBM <- predict(modGBM, testing)
    accuGBM <- confusionMatrix(predGBM, testing$classe)
    accuGBM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2192   51    0    0    2
    ##          B   32 1423   50    3   14
    ##          C    6   43 1300   46   15
    ##          D    1    0   16 1230   24
    ##          E    1    1    2    7 1387
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.96            
    ##                  95% CI : (0.9554, 0.9642)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9494          
    ##  Mcnemar's Test P-Value : 3.241e-09       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9821   0.9374   0.9503   0.9565   0.9619
    ## Specificity            0.9906   0.9844   0.9830   0.9938   0.9983
    ## Pos Pred Value         0.9764   0.9350   0.9220   0.9677   0.9921
    ## Neg Pred Value         0.9929   0.9850   0.9894   0.9915   0.9915
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2794   0.1814   0.1657   0.1568   0.1768
    ## Detection Prevalence   0.2861   0.1940   0.1797   0.1620   0.1782
    ## Balanced Accuracy      0.9863   0.9609   0.9667   0.9751   0.9801

Therefore, we see that the accuracy for modRF (random forests) is
greater than that for modGBM (boosting). We choose the method of Random
Forests for our final model fitting. To make it clear, let's have a look
at the significant contributing variables in `modRF`:

    varImp(modRF)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                   Overall
    ## roll_belt          100.00
    ## yaw_belt            78.65
    ## magnet_dumbbell_z   69.82
    ## magnet_dumbbell_y   61.04
    ## pitch_belt          60.60
    ## pitch_forearm       58.09
    ## magnet_dumbbell_x   55.75
    ## roll_forearm        52.52
    ## magnet_belt_y       45.11
    ## accel_dumbbell_y    44.85
    ## accel_belt_z        43.03
    ## roll_dumbbell       41.94
    ## magnet_belt_z       40.79
    ## accel_dumbbell_z    38.29
    ## accel_forearm_x     33.31
    ## roll_arm            33.21
    ## yaw_dumbbell        30.24
    ## accel_dumbbell_x    29.77
    ## gyros_belt_z        29.65
    ## magnet_arm_y        27.73

A plot will demonstrate the above point:

    library(ggplot2)
    p <- ggplot(data = training, aes(x=roll_belt, y=pitch_forearm, color = factor(classe))) +
            geom_point(alpha=0.4)
    p

<img src="Course8_project_files/figure-markdown_strict/unnamed-chunk-9-1.png" style="display: block; margin: auto;" />

Model validation
----------------

We use the `testingDataset` to validate the Random Forests model,
`modRF`, and get the predictions for the questions asked by the
assignment:

    predValidate <- predict(modRF, testingDataset)
    predValidate

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
