set.seed(123)
index <- createDataPartition(data$sad,p = 0.7,list = F)
train <- data[index,]
test  <- data[-index,]
#lasso and feature selection
X<-as.matrix(train[,1:53])    
Y<-as.matrix(train[,54])
lasso.cv <- cv.glmnet(x = X,y = Y,family ="binomial", alpha = 1)
lasso_min <- glmnet(  x = X,  y = Y,  family = "binomial",  alpha = 1)
lasso_lse<-lasso.cv$lambda.1se
lasso_lse
lasso.coef<-coef(lasso.cv$glmnet.fit,s=lasso_lse,exact = F)
lasso.coef
test <-test[, !(names(test) %in% c("gender","sbp","mbp","hemoglobin","cl", "k","mg","ca","p","inr"))] 
train <-train[, !(names(train) %in% c("gender","sbp","mbp","hemoglobin","cl","k","mg","ca","p","inr"))] 
#LR model
lm_model <- glm(sad ~ ., data = train, family = binomial(link = "logit"))
summary(lm_model)
lm_pred <- predict(lm_model,test, type = "response")
threshold <- 0.5
predictions_binary <- ifelse(lm_pred > threshold, 1, 0)
confusionMatrix(as.factor(predictions_binary), as.factor(test$sad))
#SVM model
svm_model <- svm(sad ~ ., data = train, probability = TRUE)
svm_pred <- predict(svm_model, test, probability = TRUE)
svm_pred_prob <- attr(svm_pred, "probabilities")[, 2]
confusionMatrix(data = factor(svm_pred, levels = c("0", "1")), reference = test$sad)
train$sad <- as.numeric(as.character(train$sad)) 
test$sad <- as.numeric(as.character(test$sad)) 
# XGBoost model
train_matrix <- xgb.DMatrix(data.matrix(train[, !names(train) %in% "sad"]), label = train$sad)
test_matrix <- xgb.DMatrix(data.matrix(test[, !names(test) %in% "sad"]), label = test$sad)
params <- list(objective = "binary:logistic",eval_metric = "logloss",max_depth = 3, eta = 0.1, 
               gamma = 0.5, colsample_bytree =1, min_child_weight = 1 ,subsample = 0.5)
watchlist <- list(train = train_matrix,val = test_matrix)
xgb_model <- xgb.train(params = params,data = train_matrix,nrounds = 125,watchlist = watchlist,
                       early_stopping_rounds = 10, print_every_n = 10,maximize = FALSE)
xgb_pred_prob <- predict(xgb_model, test_matrix)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)
xgb_pred_factor <- factor(xgb_pred, levels = c(0, 1))
test_sad_factor <- factor(test$sad, levels = c(0, 1))
confusionMatrix(data = xgb_pred_factor, reference = test_sad_factor)
# RF model
train$sad <- as.factor(train$sad)
test$sad <- as.factor(test$sad)
rf_model <- randomForest(sad ~ ., data = train, ntree = 500, mtry = 6)
rf_pred <- predict(rf_model, newdata = test)
confusionMatrix(data = rf_pred, reference = test$sad)
#DT model
dt_model <- rpart(sad ~ ., data = train, method = "class")
dt_pred_prob <- predict(dt_model, newdata = test, type = "prob")[, 2]
dt_pred <- ifelse(dt_pred_prob > 0.5, 1, 0)
confusionMatrix(factor(dt_pred, levels = c("0", "1")), test$sad)
#NB model
nb_model <- naiveBayes(sad ~ ., data = train)
nb_pred_prob <- predict(nb_model, newdata = test, type = "raw")[, 2]
nb_pred <- ifelse(nb_pred_prob > 0.5, 1, 0)
confusionMatrix(factor(nb_pred, levels = c("0", "1")), test$sad)
# KNN model
knn_model <- kknn(sad~., train, test, k = 10, distance = 2, kernel = "rectangular")
knn_pred_prob <- predict(knn_model, newdata = test, type = "prob") 
knn_pred_prob <- knn_pred_prob[, "1"]
knn_pred_prob <- as.numeric(knn_pred_prob)
threshold <- 0.5 
knn_pred <- ifelse(knn_pred_prob > threshold, 1, 0)
confusionMatrix(factor(knn_pred, levels = c("0", "1")), test$sad)
