from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from hydra import initialize, compose
from omegaconf import OmegaConf

with initialize(config_path="../config/"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)

BINS = data_cfg.final_variable.hog_bins_feature
N_CLASSES = data_cfg.final_variable.n_classes
SEED = data_cfg.final_variable.seed
NUM_TREES = data_cfg.final_variable.num_trees
MAX_DEEP = data_cfg.final_variable.max_deep


model_classifiers = [
    KNeighborsClassifier(n_neighbors=N_CLASSES),
    DecisionTreeClassifier(max_depth=5, random_state=SEED),
    RandomForestClassifier(n_estimators=NUM_TREES, max_depth=MAX_DEEP, random_state=SEED),
    SVC(kernel="linear", C=0.025, random_state=SEED),
    SVC(gamma=2, C=1, random_state=SEED),
    SVC(kernel='rbf', random_state=SEED),
    MLPClassifier(alpha=1, max_iter=1000, random_state=SEED),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    AdaBoostClassifier(random_state=SEED),
    GradientBoostingClassifier(n_estimators=NUM_TREES, learning_rate=1.0, max_depth=MAX_DEEP),
    XGBClassifier(n_estimators=NUM_TREES, random_state=SEED),
]