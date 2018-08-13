from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def get_precision_recall_curve(ys, scores, file_path):
    average_precision = average_precision_score(ys, scores)
    precision, recall, _ = precision_recall_curve(ys, scores)
    plt.step(recall)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(file_path)