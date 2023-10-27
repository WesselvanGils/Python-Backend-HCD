import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

class Model:
    def __init__(self):
        self.X_train = None
        self.features = None
        self.model = self.train()

    def train(self):
        data = pd.read_csv("./data/divorce_data.csv", sep=";")
        data = pd.concat([data.iloc[:, 0:20], data.iloc[:, -1]], axis=1) 

        X = data.drop("Divorce", axis=1).to_numpy()
        y = data["Divorce"].to_numpy()

        self.features = data.columns[:-1]

        self.X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlp = MLPRegressor(hidden_layer_sizes=(5,),activation='logistic', max_iter=10000,learning_rate='invscaling',random_state=0)
        model = make_pipeline(StandardScaler(), mlp)

        model.fit(self.X_train, y_train)
        return model

    def predict(self, data):
        data = data.reshape(1, -1)
        pred = self.model.predict(data)

        plt.clf()

        explainer = shap.KernelExplainer(self.model.predict, self.X_train[:100])
        shap_values = explainer.shap_values(data, nsamples=100)
        shap.force_plot(explainer.expected_value, shap_values, self.X_train[0,:],feature_names=self.features, matplotlib=True, show=False)

        plt.savefig('temp.png')
        plt.close()

        return pred.astype(str)