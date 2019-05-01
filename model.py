import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class Model() :
    def __init__(self) :
        self.lr = LogisticRegression(penalty='l1', C=0.1)
        self.pipeline = Pipeline(steps=[("lr", self.lr)])

    def fit(self, features, labels) :
        self.pipeline.fit(features, labels)

    def predict_proba(self, features) :
        return self.pipeline.predict_proba(features)

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
    model.fit(train_ds, train_labels)
    print("Model trained")

    predictions = model.predict_proba(test_ds)
    print("Inference done")

    test_ds["target"] = predictions[:,1]
    test_ds.to_csv(submission_file, columns=["target"])

if __name__ == "__main__" :
    main()
