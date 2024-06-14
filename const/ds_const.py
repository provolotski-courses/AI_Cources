from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.svm import SVC

output_file = 'log/output.log'
fw_output_file = 'report/dateset_output.log'
# Тренировочный набор данных
train_file4lab = 'datasets/train.csv'
# Тестовый набор данных
test_file4lab = 'datasets/test.csv'
# Название набора данных
fw_dataset_file = 'datasets/adult.csv'
# атрибут определяющий количество записей
fw_dataset_counter = 'fnlwgt'
# во сколько раз будем меньше генерировать записей
fw_dataset_divider = 120000
# словарь обучаемых моделей
fw_model_dict = {'к-ближайших':KNeighborsClassifier(n_neighbors=3),
                 'GradientBoosting': GradientBoostingClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'AdaBoost': AdaBoostClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'SVM': SVC(kernel='linear'),}



# Справочник методов с описанием для враппера
func_description = {'init_datasets': 'Загрузка датасета', 'gradient': 'Вычисление градиентного бустинга',
                    'yandex': 'Вычисление  CatBoosting', 'ada_boost': 'Вычисление Ada Boost',
                    'extra_tree': 'Вычисление Extra Trees', 'sqr_boost': 'Вычисление QDA',
                    'LGBM_method': 'Вычисление Light Gradient Boosting Machine',
                    'KNN_method': 'K ближайших соседей', 'decision_tree_method': 'Классификатор дерева решений',
                    'extremally_gradient_method': 'Экстремальный градиентный бустинг',
                    'dumn_method': 'Фиктивный классификатор', 'SVM_method': 'SVM - линейное ядро',
                    'load_dataset':'Загрузка данных','EDA_report':'Анализ датасета',
                    'analyze_pycaret':'Анализ датасета с помощью pyCaret','generate_dataframes':'Генерация наборов',
                    'train_model':'Тренировка модели'
                    }

# Справочник гиперпараметров
model_params = {'estimators': 300, 'depth': 7, 'learning_rate': 0.05, 'iterations': 300, 'verbose': 0,
                'max_features': 'sqrt', 'num_leaves': 31, 'eval_metric': 'logloss', 'strategy': "most_frequent",
                'loss_function': 'MultiClass', 'kernel': 'poly', 'n_neighbors': 6}
