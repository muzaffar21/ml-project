from text_vectorizer import Vectorizer
from classifier import Train
import operator
import sys

sys.path.append()


class load_data:
    def __init__(self):
        pass

    label_array = []
    corpus = []
    label_map = {}

    def load_data(self, training_file):
        with open(training_file, 'r') as tf:
            lines = tf.readlines()
        label_number = -1

        for line in lines:
            arr = line.split("##")
            label = arr[0].strip().lower()
            document = arr[1].strip()
            # print label
            if label not in self.label_map:
                label_number += 1
                self.label_map[label] = label_number
                self.label_array.append(label_number)
            else:
                self.label_array.append(self.label_map[label])

            self.corpus.append(document)

    def write_class_map(self, class_file):
        with open(class_file,'w') as fp:
            for label in sorted(self.label_map.items(), key=operator.itemgetter(1)):
                fp.write(str(label[0]).strip().lower() + "\n")
        fp.close()


def main(training_file, model_file, vector_file, class_file):
    ld = load_data()
    vec = Vectorizer()
    train = Train()
    ld.load_data(training_file)
    ld.write_class_map(class_file)
    print "data loaded sucessfully , going to vectorize data"
    if len(ld.label_array) != len(ld.corpus):
        print "exception occurred while loading data ::::::,size of data  set and label set un equal"
    else:
        matrix = vec.generate_vector_matrix(ld.corpus, vector_file, True)
        print 'vectors saved, going to train model'
        train.create_model(matrix, ld.label_array, model_file)
        print ".......................done training......................"


if __name__ == '__main__':
    directory = "/disk2/home/muzaffar/PycharmProjects/ski_learn/resources/"
    main(directory + "trainingdata.list",
         directory + "model.txt",
         directory + "vectorizer.pickle",
         directory + "class_file.txt")
