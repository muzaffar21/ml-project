from classifier import Predict
from text_vectorizer  import VectorizerPredict


def main(model_path,vector_path):
    print "going to load model"
    pr = Predict(model_path)
    vec = VectorizerPredict()
    vectorizer  = vec.load_vectorizer(vector_path)
    test_data = ["the service was amazing., awesome good happy ver good nice"]
    vector = vec.vectorize(test_data, vectorizer)
    print vector
    print vector
    pr.predict_label(vector)






if __name__ == '__main__':
    main("/disk2/home/muzaffar/PycharmProjects/ski_learn/model.txt",
        "/disk2/home/muzaffar/PycharmProjects/ski_learn/vectorizer.pickle" );