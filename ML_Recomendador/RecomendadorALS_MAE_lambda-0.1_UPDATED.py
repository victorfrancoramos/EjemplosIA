#!/usr/bin/env python3
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("MovieRecommenderUpdated") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load ratings data; assuming the delimiter is "::"
ratings_df = spark.read.option("delimiter", "::").csv("/user/raul.pingarron/recomendador/ratings_all.txt")
# Rename columns for clarity and cast types
ratings_df = ratings_df.selectExpr("_c0 as userId", "_c1 as movieId", "_c2 as rating") \
    .withColumn("userId", col("userId").cast("integer")) \
    .withColumn("movieId", col("movieId").cast("integer")) \
    .withColumn("rating", col("rating").cast("float"))

# Filter out rows with rating 0
ratings_df = ratings_df.filter(col("rating") != 0)

print("Ratings count:", ratings_df.count())

# Split data 70/20/10
(training_df, validation_df, test_df) = ratings_df.randomSplit([0.7, 0.2, 0.1], seed=42)

# Set up ALS model (using default parameters for starters)
als = ALS(
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating",
    coldStartStrategy="drop",  # to handle NaN predictions during evaluation
    nonnegative=True
)

# Use CrossValidator for hyperparameter tuning (optional)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [4, 8, 12, 16, 18, 20, 22]) \
    .addGrid(als.maxIter, [5, 10, 15, 20]) \
    .build()

evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel = cv.fit(training_df)

# Evaluate on the validation set
val_predictions = cvModel.transform(validation_df)
mae_valid = evaluator.evaluate(val_predictions)
print(f"Validation MAE: {mae_valid}")

# Evaluate on the test set
test_predictions = cvModel.transform(test_df)
mae_test = evaluator.evaluate(test_predictions)
print(f"Test MAE: {mae_test}")

# Generate recommendations for a specific user
# Load user ratings for "my ratings" file
my_ratings_df = spark.read.option("delimiter", "::").csv("/user/raul.pingarron/recomendador/mis_ratings.txt") \
    .toDF("userId", "movieId", "rating") \
    .withColumn("userId", col("userId").cast("integer")) \
    .withColumn("movieId", col("movieId").cast("integer")) \
    .withColumn("rating", col("rating").cast("float"))

# Get unrated movies for the user (assuming rating==0 means not rated)
unrated_df = my_ratings_df.filter(col("rating") == 0)

# Predict scores for unrated movies
recommendations = cvModel.bestModel.transform(unrated_df)
top_recs = recommendations.orderBy(col("prediction").desc()).limit(5)

# Load movies to map movieId -> title
movies_df = spark.read.option("delimiter", "::").csv("/user/raul.pingarron/recomendador/movies.dat") \
    .toDF("movieId", "title", "genre") \
    .withColumn("movieId", col("movieId").cast("integer"))

# Join recommendations with movie titles
final_recs = top_recs.join(movies_df, on="movieId", how="inner").select("title", "prediction")
print("Top 5 recommended movies:")
final_recs.show(truncate=False)

spark.stop()
