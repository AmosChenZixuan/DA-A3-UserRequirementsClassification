from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred, roundto=4):
        return round(accuracy_score(y_true, y_pred), roundto)


    @staticmethod
    def f1(y_true, y_pred, roundto=4):
        return np.around(f1_score(y_true, y_pred, average=None), roundto)

    @staticmethod
    def classification_report(y_true, y_pred):
        return classification_report(y_true, y_pred)


    @staticmethod
    def report(model, X_train, X_test, y_train, y_test):
        def run_metrics(y_true, y_pred):
            return {
                'accuracy':  Metrics.accuracy(y_true, y_pred),
                'f1':        Metrics.f1(y_true, y_pred),
            }
        yh_train = model.predict(X_train)
        train_metrics = run_metrics(y_train, yh_train)
        print(f"Train Accuracy: {train_metrics['accuracy']}, F1: {train_metrics['f1']}")
        yh_test = model.predict(X_test)
        test_metrics  = run_metrics(y_test, yh_test) 
        print(f"Test  Accuracy: {test_metrics['accuracy']}, F1: {test_metrics['f1']}")
        return train_metrics, test_metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12.8,6))
        sns.heatmap(conf_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap="Blues")

        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.title('Confusion matrix')
        plt.show()