import torch
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import MLFlowLogger

ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Tutorial1")

tokenizer = Tokenizer.load(
    pretrained_model_name_or_path="bert-base-german-cased",
    do_lower_case=False)

LABEL_LIST = ["OTHER", "OFFENSE"]
processor = TextClassificationProcessor(tokenizer=tokenizer,
    max_seq_len=128,
    data_dir="data/germeval18",
    label_list=LABEL_LIST,
    metric="f1_macro",
    label_column_name="coarse_label")

BATCH_SIZE = 32

data_silo = DataSilo(
    processor=processor,
    batch_size=BATCH_SIZE)

