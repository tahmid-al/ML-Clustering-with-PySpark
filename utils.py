from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    spark = SparkSession.builder.appName("AdvancedClustering").getOrCreate()
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    features = [col for col in df.columns if col != 'label']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    assembled = assembler.transform(df)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaled = scaler.fit(assembled).transform(assembled)
    return spark, scaled, df

def evaluate_kmeans(data, k_range=(2, 8)):
    scores = []
    for k in range(*k_range):
        kmeans = KMeans(featuresCol="scaledFeatures", k=k, seed=1)
        model = kmeans.fit(data)
        preds = model.transform(data)
        evaluator = ClusteringEvaluator()
        score = evaluator.evaluate(preds)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k, scores

def plot_silhouette_scores(scores, path):
    x, y = zip(*scores)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', color='blue')
    plt.title("Silhouette Scores for K")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(path)

def save_predictions(predictions, original_df, output_path):
    selected = predictions.select("prediction")
    combined = original_df.withColumn("cluster", selected["prediction"])
    combined.toPandas().to_csv(output_path, index=False)