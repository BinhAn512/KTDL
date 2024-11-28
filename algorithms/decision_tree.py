from sklearn.tree import DecisionTreeClassifier, export_text
import math
from collections import Counter
def decision_tree_algorithm(data):
    X = data.iloc[:, :-1]  # Đặc trưng
    y = data.iloc[:, -1]   # Nhãn
    model = DecisionTreeClassifier()
    model.fit(X, y)
    tree_rules = export_text(model, feature_names=list(X.columns))
    return f"<pre>{tree_rules}</pre>"

