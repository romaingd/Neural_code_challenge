import numpy as np

from preprocessing import load_data, preprocess_data
from classification import classify

from deeplearning.models import vggcustom, LSTMClassifier
from deeplearning.preprocessing import TimeSeriesDataset
from deeplearning.metrics import get_acc
from deeplearning.runners import run_experiment

from sklearn.model_selection import GroupShuffleSplit


# Parameters
data_folder = './data/'
isi_folder = './features/isi/'
submission_folder = './submissions/cnn/'

perform_evaluation = True

compute_submission = False



###############################################################################
#                                                                             #
#                                Data loading                                 #
#                                                                             #
###############################################################################

# Usual procedure
x_tr, x_te, y_tr = load_data(
    features_folder=isi_folder,
    data_folder=data_folder
)

preprocessing_steps = []
resampling_steps = []
x_tr, x_te, groups_tr, y_tr = preprocess_data(
    x_tr,
    x_te,
    y_tr=y_tr,
    preprocessing_steps=preprocessing_steps,
    resampling_steps=resampling_steps
)

# Time series specific preprocessing
splitter = GroupShuffleSplit(n_splits=5, test_size=0.33,
                             random_state=42)
train_idx, test_idx = next(splitter.split(x_tr, y_tr, groups_tr))

train_data = TimeSeriesDataset(x_tr.values[train_idx], y_tr.values[train_idx].ravel(), False, True)
test_data = TimeSeriesDataset(x_tr.values[test_idx], y_tr.values[test_idx].ravel(), False, False)



###############################################################################
#                                                                             #
#                                  Model run                                  #
#                                                                             #
###############################################################################


def get_models(): # tuples of (batch_size, model)
    return [
        # (1024, vggcustom(num_classes=2)),
        (256, LSTMClassifier(
            in_dim=1,
            hidden_dim=24,      # 12, 24 ~ 29.5
            num_layers=3,
            dropout=0.5,         # 0.8
            bidirectional=True,
            num_classes=2,
            batch_size=256
        )),
    ]

experiments = [
    {
        'experiment_path':'roll', 
        'train_data': train_data,
        'test_data': test_data,
        'model_root':'model', 
        'models':get_models(),
        'norm':False,
        'get_acc': get_acc,
        'resume':False,  
        'num_epoch':1000
    }
]

for experiment in experiments:
    exp_log = run_experiment(**experiment)
    print(exp_log)