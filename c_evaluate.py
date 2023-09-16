import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from inspect import signature


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))
    

def precision_recall(truth, prediction):
    tn, fp, fn, tp = confusion_matrix(truth.flatten(), prediction.flatten()).ravel()
    percision = tp/(tp+fp)
    sensitivity_or_recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    return percision, sensitivity_or_recall


def plot_precision_recall_curve(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test.flatten(), y_score.flatten())

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.figure(1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('sensitivity_or_recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.savefig("precision_recall_curve.png")
    return


def main():
    header1 = ("Dice", "Prec.", "Sens._or_Rec.", "F2")
    header2 = ("TN", "FP", "FN", "TP")
    rows1 = list()
    rows2 = list()
    subject_ids = list()
    for case_folder in glob.glob("prediction/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()

        plot_precision_recall_curve(truth, prediction)

        dice = dice_coefficient(truth, prediction)
        percision, sensitivity_or_recall = precision_recall(truth, prediction)
        f1 = (2.*percision*sensitivity_or_recall)/(percision+sensitivity_or_recall)
        f2 = (5.*percision*sensitivity_or_recall)/(4.*percision+sensitivity_or_recall)
        tn, fp, fn, tp = confusion_matrix(truth.flatten(), prediction.flatten()).ravel()

        met1 = [dice, percision, sensitivity_or_recall, f2]
        rows1.append(met1)                                 
        met2 = [tn, fp, fn, tp]
        rows2.append(met2)

    df1 = pd.DataFrame.from_records(rows1, columns=header1, index=subject_ids)
    df1.to_csv("./prediction/metrics1.csv")

    df2 = pd.DataFrame.from_records(rows2, columns=header2, index=subject_ids)
    df2.to_csv("./prediction/metrics2.csv")

    scores = dict()
    for index, score in enumerate(df1.columns):
        values = df1.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.figure(2)
    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.title("Metrics")
    plt.savefig("metrics_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.figure(3)
        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()
