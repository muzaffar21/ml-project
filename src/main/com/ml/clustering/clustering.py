__author__ = "Shah Muzaffar"

from sklearn.cluster import KMeans
import json
from src.main.com.ml.classifier.text_vectorizer import Vectorizer


def run_k_means(vec_matrix,num_clusters):
    k_means = KMeans(n_clusters=, random_state=0).fit(vec_matrix)
    clusters = k_means.labels_
    return clusters


def load_data(training_file):
    corpus = []
    with open(training_file, 'r') as tf:
        lines = tf.readlines()
    for line in lines:
        corpus.append(line.replace("\n", " ").strip())
    return corpus


def create_result_file(clusters, corpus, result_file):
    fp = open(result_file, 'w')
    if len(corpus) != len(clusters):
        print "something happened wrong coz there is size mismatch"
    else:
        x = 0
        for line in corpus:
            result_map = {'text': line, 'cluster_number': str(clusters[x])}
            fp.write(json.dumps(result_map) + "\n")
            x += 1
    fp.close()


def main(file_path, result_path,num_clusters):
    vectorizer = Vectorizer()
    corpus = load_data(file_path,num_clusters)
    vec_matrix = vectorizer.generate_vector_matrix(corpus, "not", False)
    clusters = run_k_means(vec_matrix)
    create_result_file(clusters, corpus, result_path)


if __name__ == '__main__':
    directory = "/disk2/home/muzaffar/PycharmProjects/ml-project/resources/"
    main(directory + "clustering-data.lst", directory + "clusters.json",8)
