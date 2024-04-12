#Set working directory
setwd("C:/bases/ML")

##libraries
library(dplyr)
library(BART)
library(caret)
library(ggplot2)
library(glmnet)
library(pROC)
library(doParallel)
library(randomForest)

# load datasets
load("dataV3/dataTrain.RData")
load("dataV3/dataTest.RData")
load("dataV3/dataTrain_na.RData")
load("dataV3/dataTest_na.RData")


# creating variables
x <- dataTrain_na %>% select(-c(NIPC, Year)
x.test <- dataTest_na %>% select(-c(NIPC, Year))
y <- dataTrain_na$exporterIntensity

################################
## models
###############################

### Logit 
# fit model
fit_logit <- glm(exporterIntensity ~ controls,
                 data=dataTrain_na, family=binomial())
summary(fit_logit)

x.test_logit <- x.test
predictions_logit <- predict(fit_logit, x.test_logit, type='response')

x.test_logit$predictions_logit <- predictions_logit
x.test_logit$Year <- dataTest_na$Year
x.test_logit$nipc <- dataTest_na$NIPC
x.test_logit$exporterIntensity <- dataTest_na$exporterIntensity

x.test_logit <- x.test_logit %>%
  mutate(
    expIntensity = ExporIES_euros/Total_Vendas,
    exporter = ifelse(expIntensity>0.1, 1, 0),
    test = exporterIntensity-exporter,
    predic_logit = ifelse(predictions_logit>0.5, 1, 0)
  ) %>% as_tibble()

save(x.test_logit, file="dataScores/x.test_logit.RData")

confusionMatrix(as.factor(x.test_logit$predic_logit), as.factor(x.test_logit$exporter))

# ROC analysis
roc_logit <- roc(x.test_logit, exporter, predictions_logit)
print(roc_logit)
auc(roc_logit)

plot(roc_logit)
ggroc(roc_logit, alpha = 0.5, colour = "red", linetype = 1, size = 1.5)

p <- ggplot(x.test_logit, aes(x=predictions_logit)) + 
  geom_density() + 
  geom_vline(aes(xintercept=median(predictions_logit)),
             color="blue", linetype="dashed", size=1)
plot(p)


##############
## Lasso Logit
# fit model

cross_val <- cv.glmnet(as.matrix(x), as.matrix(y), 
                       family = 'binomial', 
                       type.measure = 'class',
                       alpha = 1, 
                       nlambda = 100)

fit_min <- glmnet(as.matrix(x), as.matrix(y), 
                  family = 'binomial', 
                  alpha = 1, 
                  lambda = cross_val$lambda.min)

x.test_lasso <- x.test
predictions_lasso <- predict(fit_min, newx = as.matrix(x.test_lasso), type = 'response')
#predictions_min_coe <- predict(fit_min, newx = as.matrix(x.test), type = 'coefficients')

x.test_lasso$predictions_lasso <- predictions_lasso
x.test_lasso$exporterIntensity <- dataTest_na$exporterIntensity
x.test_lasso$ExporIES_euros <- dataTest_na$ExporIES_euros

summary(x.test_lasso$predictions_lasso)

x.test_lasso <- x.test_lasso %>%
  mutate(
    expIntensity = ExporIES_euros/Total_Vendas,
    exporter = ifelse(expIntensity>0.1, 1, 0),
    test = exporterIntensity-exporter,
    predic_lasso = ifelse(predictions_lasso>0.5, 1, 0)
  ) %>% as_tibble()

save(x.test_lasso, file="dataScores/x.test_lasso.RData")

# accuracy analysis
confusionMatrix(as.factor(x.test_lasso$predic_lasso), as.factor(x.test_lasso$exporter))

# ROC analysis
roc_lasso <- roc(x.test_lasso, exporter, predictions_lasso)
print(roc_lasso)
auc(roc_lasso)

p <- ggplot(x.test_lasso, aes(x=predictions_lasso)) + 
  geom_density() + 
  geom_vline(aes(xintercept=median(predictions_lasso)),
             color="blue", linetype="dashed", size=1)
plot(p)


####################################
##BART 
# fit model
#with token run to ensure installation works
set.seed(99)
post = lbart(
  x, y, nskip=5,ndpost=5)

# run bart model
post_bart <- pbart(x, y, sparse = T, ntree=300, numcut=600)
#post$varprob.mean>0.05
post_bart$varprob.mean

# predicting
yhat_bart <- predict(post_bart, x.test)

x.test_bart <- x.test
x.test_bart$prob_bart <- yhat_bart$prob.test.mean 
x.test_bart$ExporIES_euros <- dataTest_na$ExporIES_euros
x.test_bart$exporterIntensity <- dataTest_na$exporterIntensity 

x.test_bart <- x.test_bart %>%
  mutate(
    expIntensity = ExporIES_euros/Total_Vendas,
    exporter = ifelse(expIntensity>0.1, 1, 0),
    test = exporterIntensity-exporter,
    predic_bart = ifelse(prob_bart>0.5, 1, 0)
  ) %>% as_tibble()

#saving predictions
save(x.test_bart, file="dataScores/x.test_bart.RData")

# accuracy analysis
confusionMatrix(as.factor(x.test_bart$predic_bart), as.factor(x.test_bart$exporter))

# ROC analysis
roc_bart <- roc(x.test_bart, exporter, prob_bart)
print(roc_bart)
auc(roc_bart)

plot(roc_bart)
ggroc(roc_bart, alpha = 0.5, colour = "red", linetype = 1, size = 1.5)


####################
##BART-MIA
# Fit model
x.test <- dataTest %>% select(-c(NIPC, Year))
y <- dataTrain$exporterIntensity

x <- dataTrain 

# Register parallel backend 
nCores <- 12  # cores do PC
cl <- makeCluster(nCores)
registerDoParallel(cl)

# run bart model
post_bartM <- pbart(x, y, sparse = T, ntree=300, numcut=600)

#covariates importance analysis
post$varprob.mean>0.05
post_bartM$varprob.mean 

# predicting
yhat_bartM <- predict(post_bartM, x.test)

# Stop the parallel backend
stopCluster(cl)

png(filename="logitPlot")
plot(density(yhat_bartM$prob.test.mean), xlim = c(0.0, 1.0))
dev.off()

# creating data frame for prediction analysis
x.test_bartM <- x.test
x.test_bartM$Year <- dataTest$Year
x.test_bartM$nipc <- dataTest$NIPC
x.test_bartM$prob_bartM <- yhat_bartM$prob.test.mean 
x.test_bartM$ExporIES_euros <- dataTest$ExporIES_euros
x.test_bartM$exporterIntensity <- dataTest$exporterIntensity 

x.test_bartM <- x.test_bartM %>%
  mutate(
    expIntensity = ExporIES_euros/Total_Vendas,
    exporter = ifelse(expIntensity>0.1, 1, 0),
    test = exporterIntensity-exporter,
    predic_bartM = ifelse(prob_bartM>0.5, 1, 0)
  ) %>% as_tibble()

save(x.test_bartM, file="dataScores/x.test_bartM.RData")

# forecast analysis
confusionMatrix(as.factor(x.test_bartM$predic_bartM), as.factor(x.test_bartM$exporter))

# ROC analysis
roc_bartM <- roc(x.test_bartM, exporter, prob_bartM)
print(roc_bartM)
auc(roc_bartM)

plot(roc_bartM)
ggroc(roc_bartM, alpha = 0.5, colour = "red", linetype = 1, size = 1.5)

plot((x.test_bartM$prob))

p_bart <- ggplot(x.test_bartM, aes(x=prob)) + 
  geom_density() + 
  geom_vline(aes(xintercept=median(prob)),
             color="blue", linetype="dashed", size=1)

plot(p_bart)


#############################
# Random Forest

x <- dataTrain_na %>% select(-c(NIPC, Year)) 
x.test <- dataTest_na %>% select(-c(NIPC, Year))
y <- dataTrain_na$exporterIntensity

dataTrain_na$exporterIntensity <- factor(dataTrain_na$exporterIntensity, levels = c(0, 1))

# scale data for faster estimations
x[-3] = scale(x[-3])
x.test[-3] = scale(x.test[-3])

# fit model
classifier = randomForest(x = x[-3],
                          y = y[-3],
                          ntree = 600, random_state = 0)

y_pred = predict(classifier, newdata = x.test)
plot(density(y_pred))

x.test_RF <- x.test
x.test_RF$prob_RF <- y_pred 
x.test_RF$ExporIES_euros <- dataTest_na$ExporIES_euros
x.test_RF$exporterIntensity <- dataTest_na$exporterIntensity 

#creating df with predictions
x.test_RF <- x.test_RF %>%
  mutate(
    expIntensity = ExporIES_euros/Total_Vendas,
    exporter = ifelse(expIntensity>0.1, 1, 0),
    test = exporterIntensity-exporter,
    prob_RF = ifelse(prob_RF<0, 0, prob_RF),
    prob_RF = ifelse(prob_RF>1, 1, prob_RF),
    predic_RF = ifelse(prob_RF>0.5, 1, 0)
  ) %>% as_tibble()

# forecast analysis
confusionMatrix(as.factor(x.test_RF$predic_RF), as.factor(x.test_RF$exporter))
save(x.test_RF, file="dataScores/x.test_RF.RData")

# ROC analysis
roc_RF <- roc(x.test_RF, exporter, prob_RF)
print(roc_bartM)
auc(roc_bartM)

plot()
ggroc(roc_RF, alpha = 0.5, colour = "red", linetype = 1, size = 1.5)
