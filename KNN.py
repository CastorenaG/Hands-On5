import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importar para gráficos 3D
from sklearn import datasets

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        most_common = np.bincount(k_neighbor_labels).argmax()
        return most_common

def plot_2d_and_3d_classification(X, y, classifier, title2d="2D Classification", title3d="3D Classification"):
    fig = plt.figure(figsize=(18, 6))

    # Gráfico de los datos antes de la clasificación en 2D
    ax1 = fig.add_subplot(1, 3, 1)
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, marker='o', s=50, edgecolors='k')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title("Original Data")

    # Gráfico de la clasificación en 2D
    ax2 = fig.add_subplot(1, 3, 2)
    classified_points = classifier.predict(X)
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=classified_points, cmap=plt.cm.Spectral, marker='o', s=50, edgecolors='k')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title(title2d)

    # Gráfico de la clasificación en 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    scatter3 = ax3.scatter(X[:, 0], X[:, 1], classified_points, c=classified_points, cmap=plt.cm.Spectral, marker='o', s=50, edgecolors='k')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.set_zlabel('Class')
    ax3.set_title(title3d)

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()

# Cargar el conjunto de datos Iris de sklearn
iris = datasets.load_iris()
X_iris = iris.data[:, :2]  # Tomar solo las primeras dos características para facilitar la visualización
y_iris = iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
# Tomar el 80% de los datos como entrenamiento y el 20% como prueba
split_ratio = 0.8
split_index = int(split_ratio * len(X_iris))

X_train_iris, y_train_iris = X_iris[:split_index], y_iris[:split_index]
X_test_iris, y_test_iris = X_iris[split_index:], y_iris[split_index:]

# Crear y entrenar el clasificador con el conjunto de datos Iris
knn_classifier_iris = KNNClassifier(k=3)
knn_classifier_iris.fit(X_train_iris, y_train_iris)

# Mostrar los gráficos en 2D y 3D después de la clasificación
plot_2d_and_3d_classification(X_iris, y_iris, knn_classifier_iris, title2d="Classification (Iris)", title3d="Classification (Iris)")
