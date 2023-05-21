from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

def plot_pairwaise_class(scores, target,name = "", lt = False):
    acc = []
    dlen = scores.max() - scores.min()
    thresh = np.linspace(scores.min() - dlen / 50, scores.max() + dlen / 50, 100)
    for t in thresh:
        if lt:
            preds = scores < t
        else:
            preds = scores > t
        acc.append(accuracy_score(target, preds))
    
    fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                 ['left', 'lower right']],
                                 figsize=(18, 9), layout="constrained")
    plt.sca(axd['left'])
    sns.lineplot(x = thresh, y = acc)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.sca(axd['upper right'])
    sns.histplot(scores[target == 1], binwidth = dlen/50)
    plt.xlim((scores.min() - dlen/50, scores.max() + dlen/50))
    plt.title('true class score distr')
    plt.sca(axd['lower right'])
    sns.histplot(scores[target == 0],binwidth = dlen / 50)
    plt.xlim((scores.min() - dlen/50, scores.max() + dlen/50))
    plt.title('false class score distr')
    fig.suptitle(name)
    plt.show()
    print(f'max acc: {np.array(acc).max() :.3f}')
    print(f'class 1 prop : {target.mean()}')
    return np.array(acc).max()


def calc_sim(func, class1, class2):
    similarity_between= []
    similarity_inside = []

    for i, dial1 in tqdm(enumerate(class1)):
        for dial2 in class2:
            similarity_between.append(func(dial1, dial2))
        for j in range(i + 1, len(class1)):
            similarity_inside.append(func(dial1, class1[j]))
    return similarity_between, similarity_inside

def display_sim_distr(similarity_between, similarity_inside, is_discr = False):
    weight = [1 / len(similarity_between)] * len(similarity_between)
    weight += [1 / len(similarity_inside)] * len(similarity_inside)
    weight = np.array(weight)

    df = pd.DataFrame({'data' : similarity_between + similarity_inside, 
    'class' : ['between'] * len(similarity_between) + ['inside'] * len(similarity_inside),
    'weight' : weight})

    plt.figure(figsize = (16, 16))
    plt.subplot(2, 1, 1)
    sns.histplot(df, x = 'data', hue = 'class', weights = 'weight', 
    multiple= 'dodge', discrete = is_discr, stat = 'frequency', shrink = 0.75) 
    plt.subplot(2, 1, 2)
    sns.boxplot(df, x = 'data', y = 'class')
    plt.show()

def sim_distr_histplot(sb, si, histkwargs):
    plt.figure(figsize = (18, 9))
    plt.subplot(2, 1, 1)
    sns.histplot(sb, stat = 'probability', **histkwargs)
    plt.title('between')
    plt.subplot(2, 1, 2)
    sns.histplot(si, stat = 'probability', **histkwargs)
    plt.title('inside')
    plt.show()


def service_to_dial(data):
    unwraped = defaultdict(list)
    combined = defaultdict(list)
    for dialogue in data:
        services = dialogue.meta['services']
        for service in services:
            unwraped[service].append(dialogue)
        combined[tuple(services)].append(dialogue)
    return unwraped, combined