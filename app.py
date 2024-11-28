from flask import Flask, render_template
import pandas as pd
from algorithms.preprocessing import preprocess_data
from algorithms.apriori import apriori_algorithm
from algorithms.bayes import bayes_algorithm
from algorithms.kmeans import kmeans_algorithm
from algorithms.decision_tree import decision_tree_algorithm

app = Flask(__name__)

# Liên kết thuật toán với file dữ liệu
ALGORITHM_DATA_FILES = {
    'preprocessing': 'data/preprocessing.csv',
    'apriori': 'data/apriori.csv',
    'bayes': 'data/bayes.csv',
    'kmeans': 'data/kmeans.csv',
    'decision_tree': 'data/decision_tree.csv'
}

@app.route('/')
def index():
    return render_template('index.html', title="Trang chủ")

@app.route('/choose')
def choose_algorithm():
    return render_template('choose_algorithm.html', title="Chọn thuật toán")

@app.route('/run/<algorithm>', methods=['GET'])
def run_algorithm(algorithm):
    # Kiểm tra thuật toán hợp lệ
    if algorithm not in ALGORITHM_DATA_FILES:
        return f"Thuật toán '{algorithm}' không tồn tại.", 404

    # Đọc dữ liệu từ file tương ứng
    file_path = ALGORITHM_DATA_FILES[algorithm]
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        return f"Lỗi khi đọc dữ liệu: {e}"

    # Gọi thuật toán tương ứng
    try:
        if algorithm == 'preprocessing':
            result = preprocess_data(data)
        elif algorithm == 'apriori':
            result = apriori_algorithm(data)
        elif algorithm == 'bayes':
            result = bayes_algorithm(data)
        elif algorithm == 'kmeans':
            result = kmeans_algorithm(data)
        elif algorithm == 'decision_tree':
            result = decision_tree_algorithm(data)
        else:
            result = "Thuật toán không tồn tại."
    except Exception as e:
        return f"Lỗi khi chạy thuật toán: {e}"

    # Hiển thị kết quả
    return render_template('result.html', algorithm=algorithm, result=result, title="Kết quả")

if __name__ == '__main__':
    app.run(debug=True)
