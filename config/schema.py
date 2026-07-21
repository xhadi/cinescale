import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    EMBEDDING_DIM: int = 50
    DATASET_SIZE: str = os.getenv("DATASET_SIZE", "small").lower()
    
    # Base directories relative to project root
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    @property
    def RAW_DATA_DIR(self) -> str:
        folder_name = "ml-latest-small" if self.DATASET_SIZE == "small" else "ml-32m"
        return os.path.join(self.PROJECT_ROOT, "data", "raw", folder_name)
    
    @property
    def PROCESSED_DATA_DIR(self) -> str:
        return os.path.join(self.PROJECT_ROOT, "data", "processed")
        
    @property
    def SYNTHETIC_DATA_DIR(self) -> str:
        return os.path.join(self.PROJECT_ROOT, "data", "synthetic")
        
    @property
    def USER_FACTORS_FILENAME(self) -> str:
        return "user_factors.parquet"
        
    @property
    def MOVIE_FACTORS_FILENAME(self) -> str:
        return "movie_factors.parquet"
        
    @property
    def USER_FACTORS_PATH(self) -> str:
        return os.path.join(self.PROCESSED_DATA_DIR, self.USER_FACTORS_FILENAME)
        
    @property
    def MOVIE_FACTORS_PATH(self) -> str:
        return os.path.join(self.PROCESSED_DATA_DIR, self.MOVIE_FACTORS_FILENAME)

    @property
    def SYNTHETIC_USER_FACTORS_PATH(self) -> str:
        return os.path.join(self.SYNTHETIC_DATA_DIR, self.USER_FACTORS_FILENAME)
        
    @property
    def SYNTHETIC_MOVIE_FACTORS_PATH(self) -> str:
        return os.path.join(self.SYNTHETIC_DATA_DIR, self.MOVIE_FACTORS_FILENAME)

    @property
    def SPARK_DRIVER_MEMORY(self) -> str:
        return "8g" if self.DATASET_SIZE == "32m" else "4g"
        
    @property
    def SPARK_SHUFFLE_PARTITIONS(self) -> int:
        return 200 if self.DATASET_SIZE == "32m" else 8

    @property
    def SPARK_EXECUTOR_MEMORY(self) -> str:
        return "6g" if self.DATASET_SIZE == "32m" else "4g"

    @property
    def SPARK_DEFAULT_PARALLELISM(self) -> int:
        return 200 if self.DATASET_SIZE == "32m" else 8

# Instantiate a single configuration instance
config = AppConfig()
