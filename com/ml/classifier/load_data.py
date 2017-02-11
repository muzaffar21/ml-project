from  text_vectorizer import VectorizerTrainer
from classifier import Train


class load_data():
    label_array = []
    corpus = []

    def load_data(self, training_file):
        with open(training_file, 'r') as tf:
            lines = tf.readlines()
        label_map = {}
        label_number = -1

        for line in lines:
            arr = line.split("##")
            label = arr[0].strip().lower()
            document = arr[1].strip()
            # print label
            if label not in label_map:
                label_number += 1
                label_map[label] = label_number
                self.label_array.append(label_number)
            else:
                self.label_array.append(label_map[label])

            self.corpus.append(document)


def main(training_file, model_file, vector_file):
    ld = load_data()
    vec = VectorizerTrainer()
    train = Train()

    ld.load_data(training_file)
    print "data loaded sucessfully , going to vectorize data"
    if len(ld.label_array) != len(ld.corpus):
        print "exception occurred while loading data ::::::,size of data  set and label set un equal"
    else:
        matrix = vec.generate_vector_matrix(ld.corpus, vector_file)
        print 'vectors saved, going to train model'
        train.create_model(matrix, ld.label_array, model_file)


if __name__ == '__main__':
    main("/disk2/home/muzaffar/PycharmProjects/ski_learn/trainingdata.list",
         '/disk2/home/muzaffar/PycharmProjects/ski_learn/model.txt',
         '/disk2/home/muzaffar/PycharmProjects/ski_learn/vectorizer.pickle')
