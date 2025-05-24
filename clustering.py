import argparse
import os
from utils import load_and_prepare_data, evaluate_kmeans, plot_silhouette_scores, save_predictions

def main(k):
    print(f"🔧 Running KMeans Clustering with k={k if k else 'auto'}\n")

    # Step 1: Load and vectorize data
    spark, scaled_data, input_df = load_and_prepare_data("data/dataset.csv")

    # Step 2: If no k is given, auto-select best k using silhouette
    if not k:
        best_k, scores = evaluate_kmeans(scaled_data)
        plot_silhouette_scores(scores, "outputs/silhouette_plot.png")
        k = best_k
        print(f"✅ Best k selected using Silhouette Score: {k}")

    # Step 3: Train final model
    from pyspark.ml.clustering import KMeans
    final_kmeans = KMeans(featuresCol="scaledFeatures", k=k, seed=1)
    model = final_kmeans.fit(scaled_data)
    predictions = model.transform(scaled_data)

    # Step 4: Save results
    save_predictions(predictions, input_df, output_path="outputs/clustered_output.csv")

    print(f"\n✅ Clustering complete. Results saved to /outputs\n")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, help="Number of clusters (optional). Auto-select if not provided.")
    args = parser.parse_args()
    main(args.k)