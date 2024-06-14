import numpy
from matplotlib import pyplot as plt
from pycaret.classification import *
import pandas as pd
import seaborn as sns
from utils import util

def analyze_pycaret(dataset, target):
    log = util.DLlogger(f'report/{target}/pycharet.txt', True)
    print = log.printml
    s = setup(dataset, target=target)
    df_report = pull()
    print(df_report)
    best = compare_models()
    evaluate_model(best)
    df_report = pull()
    print(df_report)
    plot_model(best, plot='auc', save=True)
    plot_model(best, plot='confusion_matrix',save=True)
    plot_model(best, plot='class_report', save=True)
    plot_model(best, plot='boundary',save=True)
    util.move_files('AUC.png',target)
    util.move_files('Class Report.png', target)
    util.move_files('Confusion Matrix.png', target)
    util.move_files('Decision Boundary.png', target)
    predict_model(best)
    df_report = pull()
    print(df_report)



def show_histogram(dataset):

    dataset.hist(bins=20, figsize=(160,160), legend=True)
    plt.savefig('report/img/histogram.pdf')


def show_heatmap(dataset):
    corr_matrix = dataset.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig('report/img/heatmap.png')

def analyze_target(dataset,target, le_dict):
    find_key_by_value = lambda d,val: next((k for k,v in d.items() if v == val),None)
    values = dataset[target].value_counts()
    labels = [find_key_by_value(le_dict, dataset[target].unique()[iterator]) for iterator in  dataset[target].unique()]
    # Создаем круговую диаграмму
    plt.figure(figsize=(8, 8))
    plt.savefig(f'report/{target}/img/chart.png')
    plt.pie(values, labels=labels,  autopct='%1.1f%%')
    plt.title('Распределение значений')
    plt.axis('equal')  # Круговая диаграмма
    plt.savefig(f'report/{target}/img/pie_chart.png')
