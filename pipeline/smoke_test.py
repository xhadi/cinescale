import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.spark_session import get_spark_session

def run_smoke_test():
    print("Initializing Spark session...")
    spark = get_spark_session("CineScaleSmokeTest")
    
    try:
        print("\n--- Spark Session Metadata ---")
        print(f"App Name: {spark.conf.get('spark.app.name')}")
        print(f"Master: {spark.conf.get('spark.master')}")
        print(f"Spark Version: {spark.version}")
        print(f"Driver Memory: {spark.conf.get('spark.driver.memory')}")
        print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
        print("------------------------------\n")
        
        print("Creating test DataFrame...")
        data = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        schema = ["id", "name"]
        df = spark.createDataFrame(data, schema)
        
        print("Executing computation (count)...")
        count = df.count()
        print(f"DataFrame Count: {count}")
        
        print("Executing transformation (filter & show)...")
        filtered_df = df.filter(df.id > 1)
        filtered_df.show()
        
        assert count == 3, "Count mismatch!"
        assert filtered_df.count() == 2, "Filtered count mismatch!"
    finally:
        print("Stopping Spark session...")
        spark.stop()
        
    print("Smoke test completed successfully!")

if __name__ == "__main__":
    run_smoke_test()
