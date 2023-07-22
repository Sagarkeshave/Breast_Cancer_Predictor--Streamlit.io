import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, classification_report
import pickle




def create_model(data):
    X = data.drop(["diagnosis"], axis = 1)
    y = data.diagnosis

    #Scaled data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print("Accuracy score : " , accuracy_score(y_test, y_pred))
    print("Classification report :\n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv(r"C:\Users\SAGAR KESHAVE\Breast Cancer Predictor\data\data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)

    data['diagnosis'] = data["diagnosis"].map({"M":1, "B":0})
    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open("../model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("../model/test_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # with open(r"C:\Users\SAGAR KESHAVE\Breast Cancer Predictor\model\scaler.pkl", "wb") as f:
    #     pickle.dump(scaler, f)

if __name__ == "__main__":
    main()