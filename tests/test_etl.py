import os
import zipfile
from pathlib import Path
from typing import Generator
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, FloatType, LongType

from config.schema import config
from pipeline.spark_session import get_spark_session
from pipeline.etl import extract_data, transform_movies, transform_ratings, filter_low_support_items, load_data, run_etl



@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Fixture to initialize and stop the SparkSession for ETL testing."""
    spark_sess = get_spark_session("ETLTest")
    yield spark_sess
    spark_sess.stop()


def test_extract_data_existing_dir(spark: SparkSession) -> None:
    """Test loading movies and ratings from an existing raw directory."""
    raw_dir = config.RAW_DATA_DIR
    
    if not os.path.exists(raw_dir):
        pytest.skip(f"Local raw directory {raw_dir} does not exist. Skipping integration test.")
        
    movies_df, ratings_df = extract_data(spark, raw_dir)
    
    assert movies_df is not None
    assert ratings_df is not None
    
    # Check that columns are present
    assert "movieId" in movies_df.columns
    assert "title" in movies_df.columns
    assert "genres" in movies_df.columns
    
    assert "userId" in ratings_df.columns
    assert "movieId" in ratings_df.columns
    assert "rating" in ratings_df.columns
    assert "timestamp" in ratings_df.columns


def test_extract_data_zip_extraction(spark: SparkSession, tmp_path: Path) -> None:
    """Test automatic zip file extraction when the raw directory does not exist."""
    temp_raw_dir = tmp_path / "ml-latest-small"
    temp_zip_path = tmp_path / "ml-latest-small.zip"
    
    # Create dummy CSV files
    dummy_dir = tmp_path / "dummy_raw"
    dummy_dir.mkdir()
    
    movies_csv = dummy_dir / "movies.csv"
    movies_csv.write_text("movieId,title,genres\n1,Toy Story (1995),Adventure|Animation|Children|Comedy\n")
    
    ratings_csv = dummy_dir / "ratings.csv"
    ratings_csv.write_text("userId,movieId,rating,timestamp\n1,1,4.0,964982703\n")
    
    # Zip the dummy files
    with zipfile.ZipFile(temp_zip_path, 'w') as zip_ref:
        zip_ref.write(movies_csv, arcname="ml-latest-small/movies.csv")
        zip_ref.write(ratings_csv, arcname="ml-latest-small/ratings.csv")
        
    # extract_data pointing to non-existing temp_raw_dir should unpack the zip
    movies_df, ratings_df = extract_data(spark, str(temp_raw_dir))
    
    assert movies_df.count() == 1
    assert ratings_df.count() == 1
    
    assert temp_raw_dir.exists()
    assert (temp_raw_dir / "movies.csv").exists()
    assert (temp_raw_dir / "ratings.csv").exists()


def test_extract_data_not_found(spark: SparkSession, tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if neither directory nor zip exists."""
    bad_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        extract_data(spark, str(bad_dir))


def test_extract_data_invalid_zip(spark: SparkSession, tmp_path: Path) -> None:
    """Test that IOError is raised when extracting a malformed or invalid zip file."""
    temp_raw_dir = tmp_path / "ml-latest-small-bad"
    temp_zip_path = tmp_path / "ml-latest-small-bad.zip"
    
    # Write corrupt data to the zip file path
    temp_zip_path.write_text("not a real zip content")
    
    with pytest.raises(IOError):
        extract_data(spark, str(temp_raw_dir))


def test_extract_data_zip_slip_traversal(spark: SparkSession, tmp_path: Path) -> None:
    """Test that zip archives attempting directory traversal raise an IOError."""
    temp_raw_dir = tmp_path / "ml-latest-small-malicious"
    temp_zip_path = tmp_path / "ml-latest-small-malicious.zip"
    
    # Create a zip containing a member file attempting traversal (e.g. starting with ../)
    with zipfile.ZipFile(temp_zip_path, 'w') as zip_ref:
        zip_ref.writestr("../traversal_target.csv", "id,val\n")
        
    with pytest.raises(IOError) as exc_info:
        extract_data(spark, str(temp_raw_dir))
        
    assert "Attempted directory traversal in zip file" in str(exc_info.value)


def test_transform_movies(spark: SparkSession) -> None:
    """
    Test that transform_movies correctly filters null movieIds,
    removes malformed movieIds, trims and parses padded movieIds,
    casts columns to correct types, and fills null values in
    title and genres with 'Unknown'.
    """
    # Mock data with mixed cases: valid rows, null/malformed/padded movieId, null title, null genres
    data = [
        ("1", "Toy Story (1995)", "Adventure|Animation|Children|Comedy"),  # valid string id
        (None, "Null ID Movie", "Drama"),                                     # null movieId
        ("3", None, "Action"),                                                # null title
        ("4", "Null Genre Movie", None),                                      # null genres
        ("abc", "Malformed String ID", "Comedy"),                             # malformed alphabetic id
        ("12.3", "Malformed Float ID", "Drama"),                             # malformed decimal id
        ("  5  ", "Padded ID Movie", "Sci-Fi"),                               # padded string id
    ]
    schema = ["movieId", "title", "genres"]
    raw_df = spark.createDataFrame(data, schema)

    # Invoke transform_movies
    transformed_df = transform_movies(raw_df)

    # Assert schema types
    schema_fields = {field.name: field.dataType for field in transformed_df.schema.fields}
    assert isinstance(schema_fields["movieId"], IntegerType)
    assert isinstance(schema_fields["title"], StringType)
    assert isinstance(schema_fields["genres"], StringType)

    # Assert row values
    results = transformed_df.collect()
    assert len(results) == 4

    # Map by movieId for easier assertions
    results_map = {row["movieId"]: row for row in results}
    assert 1 in results_map
    assert 3 in results_map
    assert 4 in results_map
    assert 5 in results_map
    assert None not in results_map

    # Assert individual rows are cleaned and filled
    row_1 = results_map[1]
    assert isinstance(row_1["movieId"], int)
    assert isinstance(row_1["title"], str)
    assert isinstance(row_1["genres"], str)
    assert row_1["title"] == "Toy Story (1995)"
    assert row_1["genres"] == "Adventure|Animation|Children|Comedy"

    row_3 = results_map[3]
    assert isinstance(row_3["movieId"], int)
    assert isinstance(row_3["title"], str)
    assert isinstance(row_3["genres"], str)
    assert row_3["title"] == "Unknown"
    assert row_3["genres"] == "Action"

    row_4 = results_map[4]
    assert isinstance(row_4["movieId"], int)
    assert isinstance(row_4["title"], str)
    assert isinstance(row_4["genres"], str)
    assert row_4["title"] == "Null Genre Movie"
    assert row_4["genres"] == "Unknown"

    row_5 = results_map[5]
    assert isinstance(row_5["movieId"], int)
    assert isinstance(row_5["title"], str)
    assert isinstance(row_5["genres"], str)
    assert row_5["title"] == "Padded ID Movie"
    assert row_5["genres"] == "Sci-Fi"


def test_transform_ratings(spark: SparkSession) -> None:
    """
    Test that transform_ratings correctly filters out null, empty,
    or malformed ratings rows in critical columns, trims padded inputs,
    and casts columns to correct data types.
    """
    # Schema for raw ratings dataframe
    schema = ["userId", "movieId", "rating", "timestamp"]

    # Mock data to cover all specified test cases
    data = [
        # Valid rows
        ("1", "10", "4.5", "1234567890"),
        ("2", "20", "3", "1234567891"),
        
        # Null critical columns
        (None, "10", "4.5", "1234567892"),
        ("1", None, "4.5", "1234567892"),
        ("1", "10", None, "1234567892"),
        
        # Malformed userId or movieId
        ("abc", "10", "4.5", "1234567892"),
        ("1", "abc", "4.5", "1234567892"),
        ("1.5", "10", "4.5", "1234567892"),
        ("1", "10.5", "4.5", "1234567892"),
        
        # Malformed rating
        ("1", "10", "abc", "1234567892"),
        ("1", "10", "4.5.6", "1234567892"),
        
        # Padded valid inputs
        ("  3  ", "30", "5.0", "1234567893"),
        ("4", "  40  ", "  4.0  ", "1234567894"),
        
        # Padded invalid inputs
        ("  abc  ", "10", "4.5", "1234567892"),
        ("1", "  10.5  ", "4.5", "1234567892"),
        
        # Null timestamp (should be kept, with timestamp cast to null)
        ("5", "50", "3.5", None),
    ]

    raw_df = spark.createDataFrame(data, schema)

    # Invoke transform_ratings
    transformed_df = transform_ratings(raw_df)

    # Assert schema types
    schema_fields = {field.name: field.dataType for field in transformed_df.schema.fields}
    assert isinstance(schema_fields["userId"], IntegerType)
    assert isinstance(schema_fields["movieId"], IntegerType)
    assert isinstance(schema_fields["rating"], FloatType)
    assert isinstance(schema_fields["timestamp"], LongType)

    # Collect results
    results = transformed_df.collect()

    # We expect 5 valid rows:
    # 1. ("1", "10", "4.5", "1234567890") -> (1, 10, 4.5, 1234567890)
    # 2. ("2", "20", "3", "1234567891")  -> (2, 20, 3.0, 1234567891)
    # 3. ("  3  ", "30", "5.0", "1234567893") -> (3, 30, 5.0, 1234567893)
    # 4. ("4", "  40  ", "  4.0  ", "1234567894") -> (4, 40, 4.0, 1234567894)
    # 5. ("5", "50", "3.5", None) -> (5, 50, 3.5, None)
    assert len(results) == 5

    # Map by userId for validation
    results_map = {row["userId"]: row for row in results}
    
    assert 1 in results_map
    assert 2 in results_map
    assert 3 in results_map
    assert 4 in results_map
    assert 5 in results_map

    # Check individual field casting and trimming
    row_1 = results_map[1]
    assert row_1["movieId"] == 10
    assert row_1["rating"] == 4.5
    assert row_1["timestamp"] == 1234567890

    row_3 = results_map[3]
    assert row_3["movieId"] == 30
    assert row_3["rating"] == 5.0
    assert row_3["timestamp"] == 1234567893

    row_4 = results_map[4]
    assert row_4["movieId"] == 40
    assert row_4["rating"] == 4.0
    assert row_4["timestamp"] == 1234567894

    row_5 = results_map[5]
    assert row_5["movieId"] == 50
    assert row_5["rating"] == 3.5
    assert row_5["timestamp"] is None


def test_load_data(spark: SparkSession, tmp_path: Path) -> None:
    """Test that load_data correctly saves DataFrames as Parquet."""
    from typing import Any
    from unittest.mock import patch
    import pyspark.sql.readwriter
    from pyspark.sql.types import StructType, StructField

    # Patch DataFrameWriter.parquet to write via Pandas/PyArrow on Windows if winutils is missing
    def mock_parquet(writer_self: Any, path: str, *args: Any, **kwargs: Any) -> None:
        df = writer_self._df
        pdf = df.toPandas()
        os.makedirs(path, exist_ok=True)
        part_file = os.path.join(path, "part-00000-mock.parquet")
        pdf.to_parquet(part_file, index=False)
        with open(os.path.join(path, "_SUCCESS"), "w") as f:
            pass

    # Create small real DataFrames
    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True)
    ])
    movies_data = [(1, "Toy Story", "Adventure|Animation"), (2, "Jumanji", "Adventure|Fantasy")]
    movies_df = spark.createDataFrame(movies_data, movies_schema)

    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True)
    ])
    ratings_data = [(1, 1, 4.0, 964982703), (1, 2, 3.0, 964982703)]
    ratings_df = spark.createDataFrame(ratings_data, ratings_schema)

    processed_dir = str(tmp_path / "processed")

    with patch.object(pyspark.sql.readwriter.DataFrameWriter, "parquet", mock_parquet):
        movies_path, ratings_path = load_data(movies_df, ratings_df, processed_dir)

    # Verify output paths
    assert movies_path == os.path.join(processed_dir, "movies_clean.parquet")
    assert ratings_path == os.path.join(processed_dir, "ratings_clean.parquet")

    # Verify files exist
    assert os.path.exists(movies_path)
    assert os.path.exists(ratings_path)

    # Verify row counts via Pandas (since we wrote with Pandas fallback)
    movies_pdf = pd.read_parquet(os.path.join(movies_path, "part-00000-mock.parquet"))
    ratings_pdf = pd.read_parquet(os.path.join(ratings_path, "part-00000-mock.parquet"))

    assert len(movies_pdf) == 2
    assert len(ratings_pdf) == 2

    # Verify movies columns
    assert list(movies_pdf.columns) == ["movieId", "title", "genres"]

    # Verify ratings columns
    assert list(ratings_pdf.columns) == ["userId", "movieId", "rating", "timestamp"]

    # Verify movies data
    assert movies_pdf["movieId"].tolist() == [1, 2]
    assert movies_pdf["title"].tolist() == ["Toy Story", "Jumanji"]
    assert movies_pdf["genres"].tolist() == ["Adventure|Animation", "Adventure|Fantasy"]

    # Verify ratings data
    assert ratings_pdf["userId"].tolist() == [1, 1]
    assert ratings_pdf["movieId"].tolist() == [1, 2]
    assert ratings_pdf["rating"].tolist() == [4.0, 3.0]
    assert ratings_pdf["timestamp"].tolist() == [964982703, 964982703]


def test_run_etl() -> None:
    """Test the end-to-end run_etl script execution using mocks."""
    from unittest.mock import MagicMock, patch
    
    # Mock get_spark_session, extract_data, transform_movies, transform_ratings, and load_data
    mock_spark = MagicMock()
    mock_movies_df = MagicMock()
    mock_ratings_df = MagicMock()
    mock_clean_movies_df = MagicMock()
    mock_clean_ratings_df = MagicMock()
    
    mock_movies_df.count.return_value = 10
    mock_ratings_df.count.return_value = 20
    mock_clean_movies_df.count.return_value = 8
    mock_clean_ratings_df.count.return_value = 18
    
    with patch("pipeline.etl.get_spark_session", return_value=mock_spark) as mock_get_session, \
         patch("pipeline.etl.extract_data", return_value=(mock_movies_df, mock_ratings_df)) as mock_extract, \
         patch("pipeline.etl.transform_movies", return_value=mock_clean_movies_df) as mock_trans_movies, \
         patch("pipeline.etl.transform_ratings", return_value=mock_clean_ratings_df) as mock_trans_ratings, \
         patch("pipeline.etl.filter_low_support_items", return_value=mock_clean_ratings_df) as mock_filter, \
         patch("pipeline.etl.load_data", return_value=("movies_path", "ratings_path")) as mock_load:
         
        run_etl()
        
        # Verify functions were called correctly
        mock_get_session.assert_called_once_with("CineScaleETL")
        mock_extract.assert_called_once_with(mock_spark, config.RAW_DATA_DIR)
        mock_trans_movies.assert_called_once_with(mock_movies_df)
        mock_trans_ratings.assert_called_once_with(mock_ratings_df)
        mock_filter.assert_called_once_with(mock_clean_ratings_df, min_ratings=10)
        mock_load.assert_called_once_with(mock_clean_movies_df, mock_clean_ratings_df, config.PROCESSED_DATA_DIR)
        mock_spark.stop.assert_called_once()


def test_run_etl_exception_stops_spark() -> None:
    """Test that Spark session is stopped even when an exception occurs in run_etl."""
    from unittest.mock import MagicMock, patch
    
    mock_spark = MagicMock()
    
    # Patch get_spark_session to return mock_spark, and extract_data to raise exception
    with patch("pipeline.etl.get_spark_session", return_value=mock_spark) as mock_get_session, \
         patch("pipeline.etl.extract_data", side_effect=ValueError("Test exception")) as mock_extract:
         
        with pytest.raises(ValueError, match="Test exception"):
            run_etl()
            
        # Verify that Spark was stopped in finally block
        mock_get_session.assert_called_once_with("CineScaleETL")
        mock_spark.stop.assert_called_once()


def test_etl_e2e(spark: SparkSession, tmp_path: Path) -> None:
    """End-to-end integration test using the real MovieLens small dataset."""
    from typing import Any
    from unittest.mock import patch
    import pyspark.sql.readwriter
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType, StructField

    raw_dir = config.RAW_DATA_DIR
    if not os.path.exists(raw_dir):
        pytest.skip(f"Local raw directory {raw_dir} does not exist. Skipping integration test.")

    # Patch DataFrameWriter.parquet to write via Pandas/PyArrow on Windows if winutils is missing
    def mock_parquet(writer_self: Any, path: str, *args: Any, **kwargs: Any) -> None:
        # Accessing protected member _df is acceptable for testing, but let's annotate it
        df: DataFrame = writer_self._df
        pdf = df.toPandas()
        os.makedirs(path, exist_ok=True)
        part_file = os.path.join(path, "part-00000-mock.parquet")
        pdf.to_parquet(part_file, index=False)
        with open(os.path.join(path, "_SUCCESS"), "w") as f:
            pass

    def load_parquet_with_fallback(path: str, schema: StructType) -> DataFrame:
        """Helper to read parquet natively, or fallback to Pandas/PyArrow on Windows."""
        try:
            return spark.read.parquet(path)
        except Exception as e:
            # Fallback for Windows environment without winutils
            # We catch Exception specifically to handle Py4J / winutils execution errors
            part_file = os.path.join(path, "part-00000-mock.parquet")
            if not os.path.exists(part_file):
                raise FileNotFoundError(f"Mock parquet part file not found at {part_file}") from e
            pdf = pd.read_parquet(part_file)
            return spark.createDataFrame(pdf, schema=schema)

    # Use patch.object to mock parquet writing
    with patch.object(pyspark.sql.readwriter.DataFrameWriter, "parquet", mock_parquet):
        raw_movies_df, raw_ratings_df = extract_data(spark, raw_dir)
        raw_movies_count = raw_movies_df.count()
        raw_ratings_count = raw_ratings_df.count()

        movies_clean_df = transform_movies(raw_movies_df)
        ratings_clean_df = transform_ratings(raw_ratings_df)

        clean_movies_count = movies_clean_df.count()
        clean_ratings_count = ratings_clean_df.count()

        processed_dir = str(tmp_path / "processed")
        movies_path, ratings_path = load_data(movies_clean_df, ratings_clean_df, processed_dir)

    assert os.path.exists(movies_path)
    assert os.path.exists(ratings_path)

    # Verify row counts match source files (MovieLens Small has clean data)
    assert raw_movies_count == 9742, f"Expected 9742 raw movies, got {raw_movies_count}"
    assert raw_ratings_count == 100836, f"Expected 100836 raw ratings, got {raw_ratings_count}"
    assert clean_movies_count == 9742, f"Expected 9742 clean movies, got {clean_movies_count}"
    assert clean_ratings_count == 100836, f"Expected 100836 clean ratings, got {clean_ratings_count}"
    assert clean_movies_count == raw_movies_count, "Clean movies should equal raw movies for clean data"
    assert clean_ratings_count == raw_ratings_count, "Clean ratings should equal raw ratings for clean data"

    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True)
    ])
    movies_read_df = load_parquet_with_fallback(movies_path, movies_schema)

    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True)
    ])
    ratings_read_df = load_parquet_with_fallback(ratings_path, ratings_schema)

    expected_movies_schema = {"movieId": "int", "title": "string", "genres": "string"}
    expected_ratings_schema = {"userId": "int", "movieId": "int", "rating": "float", "timestamp": "bigint"}

    actual_movies_schema = {field.name: field.dataType.simpleString() for field in movies_read_df.schema.fields}
    actual_ratings_schema = {field.name: field.dataType.simpleString() for field in ratings_read_df.schema.fields}

    assert actual_movies_schema == expected_movies_schema
    assert actual_ratings_schema == expected_ratings_schema




