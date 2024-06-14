from sklearn.metrics import accuracy_score, classification_report
from utils.util import time_logger

@time_logger
def train_model(model_classifier, model_x_train, model_y_train, model_x_test, model_y_test, method):
    """Обучение модели
    :param method: наименование метода
    :param model_y_test: проверочный целевой параметр
    :param model_y_train: тренировочный целевой параметр
    :param model_x_test: проверочный датасет
    :param model_x_train: тренировочный датасет
    :param model_classifier: передаваемый классификатор

    """
    clf = model_classifier
    clf.fit(model_x_train, model_y_train)
    y_pred = clf.predict(model_x_test)  # Считаем предсказания
    # Оценка модели
    accuracy = accuracy_score(model_y_test, y_pred)
    report = classification_report(model_y_test, y_pred)
    print(f' Report for {method} method.')
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)
def gen_rep():
    pass