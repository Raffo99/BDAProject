import pandas as pd
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, DecimalType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LinearSVC, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession\
            .builder\
            .appName("TwitterClassification")\
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
    [StructField("Buzz", IntegerType(), True)]
    )
    
    twitter_classification = spark.read.format("csv")\
      .schema(twitter_schema)\
      .option("sep", ",")\
      .option("path","/user/ubuntu/data/twitter_classification.csv").load()
    
    # Process dataset for spark
    vec_assembler = VectorAssembler(inputCols=twitter_classification.columns[:-1], outputCol="features")
    stages = [vec_assembler]
    pipeline = Pipeline(stages = stages)
    pipeline_mod = pipeline.fit(twitter_classification)
    twitter_transfd = pipeline_mod.transform(twitter_classification)
    inp_twitter_classification = twitter_transfd.select(['features', 'Buzz'])  
    
    # Split dataset in train and test set
    train_twitter_classification, test_twitter_classification = inp_twitter_classification.randomSplit([0.75, 0.25], seed=0)  

    # Creation of BinaryClassificationEvaluator object for evaluation of predictions
    classification_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="Buzz", metricName="areaUnderROC")
    
    # Logistic Regression
    logistic_regression = LogisticRegression(featuresCol='features', labelCol='Buzz', maxIter=1000)
    logistic_regression_model = logistic_regression.fit(train_twitter_classification)
    logistic_regression_preds_train = logistic_regression_model.transform(train_twitter_classification)
    logistic_regression_preds_test  = logistic_regression_model.transform(test_twitter_classification)
    print("Logistic Regression AuROC (train):", classification_eval.evaluate(logistic_regression_preds_train))
    print("Logistic Regression AuROC (test):", classification_eval.evaluate(logistic_regression_preds_test))    
    
    # Linear SVC
    svc = LinearSVC(featuresCol='features', labelCol='Buzz', maxIter=1000)
    svc_model = svc.fit(train_twitter_classification)
    svc_preds_train = svc_model.transform(train_twitter_classification)
    svc_preds_test  = svc_model.transform(test_twitter_classification)
    print("Linear SVC AuROC (train):", classification_eval.evaluate(svc_preds_train))
    print("Linear SVC AuROC (test):", classification_eval.evaluate(svc_preds_test))
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier(featuresCol='features', labelCol='Buzz', maxDepth=3)
    decision_tree_model = decision_tree.fit(train_twitter_classification)
    dt_preds_train = decision_tree_model.transform(train_twitter_classification)
    dt_preds_test  = decision_tree_model.transform(test_twitter_classification)
    print("Decision Tree AuROC (train):", classification_eval.evaluate(dt_preds_train))
    print("Decision Tree AuROC (test):", classification_eval.evaluate(dt_preds_test))