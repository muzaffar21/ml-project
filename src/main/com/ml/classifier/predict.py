# coding=utf-8

import classifier
import text_vectorizer


def load_class_map(class_file):
    class_map = {}
    x = 0
    with open(class_file) as f:
        for line in f.readlines():
            class_map[x] = line
            x += 1
    return class_map


def main(model_path, vector_path, class_file):
    print "going to load model"
    class_map = load_class_map(class_file)
    model = classifier.load(model_path)
    vectorizer = text_vectorizer.load_vectorizer(vector_path)
    test_data = ["Looks nice and beautiful and very good ", "this is bad"]
    vector = text_vectorizer.vectorize(test_data, vectorizer)
    i = 0
    label_list = list(classifier.predict_label(model, vector))
    for lbl in label_list:
        print "class predicted for \"" + str(test_data[i]) + "\" is " + str(class_map[lbl])
        i += 1


if __name__ == '__main__':
    directory = "/disk2/home/muzaffar/PycharmProjects/ml-project/resources/"
    main(directory + "model.txt",
         directory + "vectorizer.pickle",
         directory + "class_file.txt")
