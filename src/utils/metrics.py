import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def classification_report_pd(y_test, y_pred):
    report = pd.DataFrame(classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    report.support = report.support.astype(int)
    report.loc['accuracy', 'support'] = report.loc['macro avg', 'support']
    report.loc['accuracy', 'precision'] = np.nan
    report.loc['accuracy', 'recall'] = np.nan
    return report