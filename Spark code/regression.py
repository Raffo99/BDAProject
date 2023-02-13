import pandas as pd
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, DecimalType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

if __name__ == "__main__":
    spark = SparkSession\
            .builder\
            .appName("TwitterRegression")\
            .getOrCreate()
            
    twitter_schema = StructType([StructField("NCD_" + str(index), IntegerType(), True) for index in range(7)] +
    [StructField("AI_" + str(index), IntegerType(), True) for index in range(7)] +
    [StructField("AS(NA)_" + str(index), DoubleType(), True) for index in range(7)] +
    [StructField("BL_" + str(index), DoubleType(), True) for index in range(7)] +
    [StructField("NAC_" + str(index), IntegerType(), True) for index in range(7)] +
    [StructField("AS(NAC)_" + str(index), DoubleType(), True) for index in range(7)] +
    [StructField("CS_" + str(index), DoubleType(), True) for index in range(7)] +
    [StructField("AT_" + str(index), DoubleType(), True) for index in range(7)] +
    [StructField("NA_" + str(index), IntegerType(), True) for index in range(7)] +
    [StructField("ADL_" + str(index), DoubleType(), True) for index in range(7)] +
    [StructField("NAD_" + str(index), IntegerType(), True) for index in range(7)] +
    [StructField("Buzz", DoubleType(), True)]
    )

    twitter = spark.read.format("csv")\
      .schema(twitter_schema)\
      .option("sep", ",")\
      .option("path","/user/ubuntu/data/twitter_regression.csv").load()
    
    # Process dataset for spark
    vec_assembler = VectorAssembler(inputCols=twitter.columns[:-1], outputCol="features")
    stages = [vec_assembler] #, mmScaler]
    pipeline = Pipeline(stages = stages)
    pipeline_mod = pipeline.fit(twitter)
    twitter_transfd = pipeline_mod.transform(twitter)
    inp_twitter = twitter_transfd.select(['features', 'Buzz'])
    
    
    # Split dataset in train and test set
    train_twitter, test_twitter = inp_twitter.randomSplit([0.75, 0.25], seed=0)
    _, cv_set = inp_twitter.randomSplit([0.9, 0.1], seed=0)
    
    # Creation of RegressionEvaluator object for evaluation of predictions
    regr_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Buzz", metricName="r2")
    
    # Linear Regression
    linear_regression = LinearRegression(featuresCol='features', labelCol='Buzz', elasticNetParam=0.1, regParam=0.1)
    cross_val = CrossValidator(estimator=linear_regression, 
                           evaluator=regr_eval,
                           estimatorParamMaps=ParamGridBuilder().build(),
                           numFolds=5)
    cross_val_model = cross_val.fit(cv_set)
    
    lr_cv_avg = cross_val_model.avgMetrics[0]
    print("Linear Regression Cross Validation Avg R2: ", lr_cv_avg)
    
    linear_regression = LinearRegression(featuresCol='features', labelCol='Buzz', elasticNetParam=0.1, regParam=0.1)
    lr_model = linear_regression.fit(train_twitter)
    
    lr_preds_train = lr_model.transform(train_twitter)
    lr_preds_test = lr_model.transform(test_twitter)
    print("Linear Regression R2 (train):", regr_eval.evaluate(lr_preds_train))
    print("Linear Regression R2 (test):", regr_eval.evaluate(lr_preds_test))
    
    # Ridge Regression
    ridge_regression = LinearRegression(featuresCol='features', labelCol='Buzz', elasticNetParam=0, regParam=0.1)
    cross_val = CrossValidator(estimator=ridge_regression, 
                           evaluator=regr_eval,
                           estimatorParamMaps=ParamGridBuilder().build(),
                           numFolds=5)
    cross_val_model = cross_val.fit(cv_set)
    
    rr_cv_avg = cross_val_model.avgMetrics[0]
    print("Ridge Regression Cross Validation Avg R2: ", rr_cv_avg)
    
    ridge_regression = LinearRegression(featuresCol='features', labelCol='Buzz', elasticNetParam=0, regParam=0.1)
    rr_model = ridge_regression.fit(train_twitter)
    
    rr_preds_train = rr_model.transform(train_twitter)
    rr_preds_test = rr_model.transform(test_twitter)

    print("Ridge Regression R2 (train):", regr_eval.evaluate(rr_preds_train))
    print("Ridge Regression R2 (test):", regr_eval.evaluate(rr_preds_test))
    
    # Lasso Regression
    lasso_regression = LinearRegression(featuresCol='features', labelCol='Buzz', elasticNetParam=1, regParam=0.1)
    cross_val = CrossValidator(estimator=lasso_regression, 
                           evaluator=regr_eval,
                           estimatorParamMaps=ParamGridBuilder().build(),
                           numFolds=5)
    cross_val_model = cross_val.fit(cv_set)
    
    lasso_cv_avg = cross_val_model.avgMetrics[0]
    print("Lasso Regression Cross Validation Avg R2: ", lasso_cv_avg)
    
    lasso_regression = LinearRegression(featuresCol='features', labelCol='Buzz', elasticNetParam=1, regParam=0.1)
    lasso_model = lasso_regression.fit(train_twitter)

    lasso_preds_train = lasso_model.transform(train_twitter)
    lasso_preds_test = lasso_model.transform(test_twitter)

    print("Lasso Regression R2 (train):", regr_eval.evaluate(lasso_preds_train))
    print("Lasso Regression R2 (test):", regr_eval.evaluate(lasso_preds_test))