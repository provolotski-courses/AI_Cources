
output_file = 'log/output.log'
# Тренировочный набор данных
train_file = 'datasets/train.csv'
# Тестовый набор данных
test_file = 'datasets/test.csv'
# Справочник методов с описанием для враппера
func_description = {'init_datasets': 'Загрузка датасета', 'gradient': 'Вычисление градиентного бустинга',
                    'yandex': 'Вычисление  CatBoosting', 'ada_boost': 'Вычисление Ada Boost',
                    'extra_tree': 'Вычисление Extra Trees', 'sqr_boost': 'Вычисление QDA',
                    'LGBM_method': 'Вычисление Light Gradient Boosting Machine',
                    'KNN_method': 'K ближайших соседей', 'decision_tree_method': 'Классификатор дерева решений',
                    'extremally_gradient_method': 'Экстремальный градиентный бустинг',
                    'dumn_method': 'Фиктивный классификатор', 'SVM_method': 'SVM - линейное ядро'
                    }

#Справочник гиперпараметров
model_params = {'estimators': 300, 'depth': 7, 'learning_rate': 0.05, 'iterations': 300, 'verbose': 0,
                    'max_features': 'sqrt', 'num_leaves': 31, 'eval_metric': 'logloss', 'strategy': "most_frequent",
                    'loss_function': 'MultiClass', 'kernel': 'poly', 'n_neighbors': 6}

