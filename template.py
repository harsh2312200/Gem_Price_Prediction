import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "src/gem_price_prediction/__init__.py",
    "src/gem_price_prediction/components/__init__.py",
    "src/gem_price_prediction/components/data_ingestion.py",
    "src/gem_price_prediction/components/data_transformation.py",
    "src/gem_price_prediction/components/model_trainer.py",
    "src/gem_price_prediction/components/model_evaluation.py",
    "src/gem_price_prediction/pipeline/__init__.py",
    "src/gem_price_prediction/pipeline/training_pipeline.py",
    "src/gem_price_prediction/pipeline/prediction_pipeline.py",
    "src/gem_price_prediction/utils/__init__.py",
    "src/gem_price_prediction/utils/utils.py",
    "src/gem_price_prediction/logger/__init__.py",
    "src/gem_price_prediction/logger/logger.py",
    "src/gem_price_prediction/exception/__init__.py",
    "src/gem_price_prediction/exception/exception.py",
    "experiments/notebooks/experiments.ipynb",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir,file_name=os.path.split(filepath)
    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open (filepath,"w") as f:
            pass
