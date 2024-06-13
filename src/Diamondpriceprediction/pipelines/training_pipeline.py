import os
import sys
import pandas as pd
# from src.Diamondpriceprediction.logger import logging
# from src.Diamondpriceprediction.exception import customexception
from src.Diamondpriceprediction.components.data_ingestion import DataIngestion
from src.Diamondpriceprediction.components.data_transformation import DataTransformation
from src.Diamondpriceprediction.components.model_trainer import ModelTrainer


obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

model_trainer_obj = ModelTrainer()

model_trainer_obj.initate_model_training(train_arr, test_arr)

# o/p
# DELL@SachinKPC MINGW64 /d/Mytech/IDE_workspace/ksachin5136git/DiamondPricePredictionProject_repo/DiamondPricePredictionProject (main)
# $ python ./src/Diamondpriceprediction/pipelines/training_pipeline.py
# {'LinearRegression': 0.9367771815958046, 'Lasso': 0.9367605562222925, 'Ridge': 0.9367791230964326, 'Elasticnet': 0.8570807193720323}

# ====================================================================================

# Best Model Found , Model Name : Ridge , R2 Score : 0.9367791230964326

# ====================================================================================

