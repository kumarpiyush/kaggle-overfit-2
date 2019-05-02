import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

EPS = 0.1**7

class Model() :
    def __init__(self) :
        self.lr = LogisticRegression(penalty='l1', C=0.1)
        self.svm = SVC(C=2, probability=True)

    def get_consequential_features(self, features, labels) :
        self.lr.fit(features, labels)

        cons_features = []

        for i in range(len(self.lr.coef_[0])) :
            f = self.lr.coef_[0][i]
            if f > EPS or f < -EPS :
                cons_features.append(str(i))

        return cons_features

    def fit(self, features, labels) :
        self.svm.fit(features, labels)

    def predict_proba(self, features) :
        return self.svm.predict_proba(features)

def main() :
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    submission_file = sys.argv[3]

    train_ds = pd.read_csv(train_file)
    test_ds = pd.read_csv(test_file)
    train_ds = train_ds.set_index("id", drop=True)
    test_ds = test_ds.set_index("id", drop=True)

    train_labels = train_ds["target"]
    train_ds = train_ds.drop("target", axis=1)

    print("Preprocessing done")

    model = Model()

    cons_features = model.get_consequential_features(train_ds, train_labels)
    train_ds = train_ds[cons_features]
    test_ds = test_ds[cons_features]

    model.fit(train_ds, train_labels)
    print("Model trained")

    predictions = model.predict_proba(test_ds)
    print("Inference done")

    test_ds["target"] = predictions[:,1]
    test_ds.to_csv(submission_file, columns=["target"])

if __name__ == "__main__" :
    main()
