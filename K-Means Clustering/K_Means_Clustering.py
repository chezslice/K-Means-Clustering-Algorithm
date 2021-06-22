import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Load the data using sklearn module and scale down the data.
digits = load_digits()
data = scale(digits.data)

# Converting features into a range between 1 and -1 simpliftying the calculations make the data easier and more accurate data representation.
# The amount of clusters is set to 10. Defining the amount of samples and features to get the shap of the data.
k = 10
samples, features = data.shape

# Scoring the data using the sklearn module documentation. Computes many different scores for different part of the data in the model.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# Training the model by creating a K Means classifier passing the classifer thru the bench_k_means function and finally scoring the data.
clf = KMeans(n_clusters=k, init="random", n_init=10)
k_means(clf, "1", data)






