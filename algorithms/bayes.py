from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def bayes_algorithm(data):
    # Giả định cột cuối cùng là nhãn (label)
    X = data.iloc[:, :-1]  # Các cột đặc trưng
    y = data.iloc[:, -1]   # Cột nhãn
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tạo và huấn luyện mô hình
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return f"Độ chính xác: {accuracy:.2f}"

    # return classification_report(y_test, ac, output_dict=False)

