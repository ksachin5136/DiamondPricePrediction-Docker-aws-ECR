import os
import sys
import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer


obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr, test_arr = data_transformation.initialize_data_transformation(
    train_data_path, test_data_path)

model_trainer_obj = ModelTrainer()

model_trainer_obj.initate_model_training(train_arr, test_arr)

# o/p
# DELL@SachinKPC MINGW64 /d/Mytech/IDE_workspace/ksachin5136git/DimondPricePredictionProject_repo/DimondPricePredictionProject (main)
# $ python ./src/DimondPricePrediction/pipelines/training_pipeline.py
# {'LinearRegression': 0.9369214171493936, 'Lasso': 0.9368930644634293, 'Ridge': 0.9369230536085107, 'Elasticnet': 0.8560811698986379}

# ====================================================================================

# Best Model Found , Model Name : Ridge , R2 Score : 0.9369230536085107

# ====================================================================================
