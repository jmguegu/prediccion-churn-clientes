import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def preparar_datos(df, target):
    """
    Separa el dataset en features (X) y target (y), y realiza un train-test split.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrenar_modelo(X_train, y_train):
    """
    Entrena un modelo Random Forest sobre los datos de entrenamiento.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluar_modelo(clf, X_test, y_test):
    """
    Evalúa el modelo: muestra matriz de confusión y clasificación.
    """
    y_pred = clf.predict(X_test)
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

