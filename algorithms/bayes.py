import pandas as pd
import numpy as np

def bayes_algorithm(data):
    # Tách dữ liệu và nhãn
    X = data.iloc[:, :-1].values  # Các đặc trưng
    y = data.iloc[:, -1].values   # Nhãn

    # Tính toán các xác suất của các lớp (P(Class))
    classes = np.unique(y)  # Lấy các lớp duy nhất
    class_probs = {}
    for c in classes:
        class_probs[c] = np.sum(y == c) / len(y)
    
    # Tính toán các xác suất có điều kiện (P(Feature|Class))
    feature_probs = {}
    for c in classes:
        # Lọc các dữ liệu của lớp c
        class_data = X[y == c]
        feature_probs[c] = {}
        for i in range(X.shape[1]):
            # Tính toán xác suất cho mỗi đặc trưng với giả định là phân phối chuẩn
            feature_probs[c][i] = {
                'mean': np.mean(class_data[:, i]),
                'std': np.std(class_data[:, i])
            }
    
    # Hàm tính xác suất điều kiện P(Feature|Class)
    def calculate_conditional_prob(x, c):
        prob = 1.0
        for i in range(len(x)):
            mean = feature_probs[c][i]['mean']
            std = feature_probs[c][i]['std']
            # Tính xác suất theo phân phối chuẩn
            prob *= (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x[i] - mean) ** 2 / std ** 2))
        return prob
    
    # Dự đoán lớp cho dữ liệu mới
    def predict(x):
        posteriors = {}
        for c in classes:
            # Tính xác suất hậu nghiệm P(Class|Features) = P(Features|Class) * P(Class)
            posterior = class_probs[c] * calculate_conditional_prob(x, c)
            posteriors[c] = posterior
        # Chọn lớp có xác suất hậu nghiệm cao nhất
        return max(posteriors, key=posteriors.get)

    # Dự đoán cho tất cả các mẫu trong tập dữ liệu
    predictions = [predict(x) for x in X]
    
    # Tính độ chính xác
    accuracy = np.mean(predictions == y)
    return f"Độ chính xác: {accuracy:.2f}"

