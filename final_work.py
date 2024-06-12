import pandas as pd
from sklearn.preprocessing import LabelEncoder

import const.ds_const as CONST

le_dict = {}


def load_dataset(filename):
    """Загрузка датасета из файла"""
    # загружаем датасет
    loaded_dataset = pd.read_csv(filename)
    # в представленном датасете одно из полей означает сколько было людей с таким набором атрибутов
    # поэтому повторяем каждую строку столько раз сколько необходимо
    # с учетом отсутсвия необходимости имитировать перепись США количество повторов уменьшаем в нужное количество раз
    loaded_dataset = loaded_dataset.loc[loaded_dataset.index.repeat(loaded_dataset[CONST.fw_dataset_counter] / CONST.fw_dataset_divider)].reset_index(drop=True)
    #удаляем столбец определяющий количество повторений
    loaded_dataset = loaded_dataset.drop(columns=[CONST.fw_dataset_counter])
    # перемешиваем датасет
    loaded_dataset = loaded_dataset.sample(frac=1)
    return loaded_dataset


def EDA_report(analyzed_dataset):
    global le_dict
    """Проведение EDA-анализа над переданным датасетом"""
    """Вывод статистики по дата фрейму"""

    print('***---- head  ------***')
    print(analyzed_dataset.head())

    print('***---- shape  ------***')
    print(analyzed_dataset.shape)

    print('***---- нулевые значения  ------***')
    print(analyzed_dataset.isnull().sum())
    print('***---- Анализ атрибутов  ------***')
    le = LabelEncoder()
    for col in analyzed_dataset.columns:
        print(f'проводим разбор по атрибуту {col}')
        print(f'тип данных: {analyzed_dataset[col].dtypes }')
        print(f'гранулярность данных {analyzed_dataset[col].nunique()}')
        # if analyzed_dataset[col].nunique()<10:
        print('Частота появления каждого значения:')
        print(analyzed_dataset[col].value_counts())
        if analyzed_dataset[col].dtypes!='int64':
            analyzed_dataset.iloc[:, analyzed_dataset.columns.get_loc(col)] = le.fit_transform(analyzed_dataset.iloc[:, analyzed_dataset.columns.get_loc(col)])
            le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            print(analyzed_dataset[col].value_counts())


        print (le_dict)




if __name__ == '__main__':
    census_dataset = load_dataset(CONST.fw_dataset_file)
    EDA_report(census_dataset)
