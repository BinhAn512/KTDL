from flask import Flask, render_template, request
import pandas as pd
from algorithms.preprocessing import preprocess_data
from algorithms.apriori import apriori_algorithm
from algorithms.bayes import bayes_algorithm
from algorithms.k_means import kmeans_algorithm
from algorithms.decision_tree import decision_tree_algorithm
from algorithms.kohonen import kohonen_algorithm


app = Flask(__name__)

# Liên kết thuật toán với file dữ liệu
ALGORITHM_DATA_FILES = {
    'preprocessing': 'data/preprocessing.csv',
    'apriori': 'data/apriori.csv',
    'bayes': 'data/bayes.csv',
    'k-means': 'data/kmeans.csv',
    'decision_tree': 'data/decision_tree.csv',
    'kohonen': 'data/kohonen.csv'  
}


@app.route('/')
def index():
    return render_template('index.html', title="Trang chủ")

@app.route('/choose')
def choose_algorithm():
    return render_template('choose_algorithm.html', title="Chọn thuật toán")

@app.route('/upload_csv', methods=['GET'])
def upload_csv():
    algorithm = request.args.get('algorithm')  # Lấy thuật toán từ query string
    if not algorithm:
        return "Bạn chưa chọn thuật toán.", 400
    return render_template('upload_csv.html', algorithm=algorithm, title="Tải lên CSV")

@app.route('/process_csv/<algorithm>', methods=['POST'])
def process_csv(algorithm):
    file = request.files.get('file')  # Lấy file CSV
   

    if not file:
        return "Bạn chưa tải lên file CSV.", 400

    try:
        # Đọc dữ liệu từ file CSV
        data = pd.read_csv(file)

        # Chuyển đổi dữ liệu gốc sang HTML để hiển thị
        data_html = data.to_html(classes="table table-striped", index=False)

        # Chạy thuật toán tương ứng
        if algorithm == 'preprocessing':
            result = preprocess_data(data)
            image_path = None
        if algorithm == 'apriori':
            min_support = float(request.form.get('min_support', 0.3))
            min_confidence = float(request.form.get('min_confidence', 0.7))
            result = apriori_algorithm(data, min_support=min_support, min_confidence=min_confidence)
            image_path = None
        elif algorithm == 'bayes':
            result = bayes_algorithm(data)
            image_path = None
        elif algorithm == 'k-means':
            result, image_path = kmeans_algorithm(data)
        elif algorithm == 'decision_tree':
            result, image_path = decision_tree_algorithm(data)
        elif algorithm == 'kohonen':
            result, image_path = kohonen_algorithm(data)
        else:
            return "Thuật toán không hợp lệ.", 400
    except Exception as e:
        return f"Lỗi khi xử lý file hoặc chạy thuật toán: {e}"

    # Hiển thị kết quả
    return render_template(
        'result.html',
        algorithm=algorithm,
        data_html=data_html,  # Truyền bảng dữ liệu CSV
        result=result,
        image_path=image_path,
        title="Kết quả"
    )


    
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            data = pd.read_csv(file)
            try:
                # Gọi hàm KMeans và nhận kết quả
                result, image_path = kmeans_algorithm(data)
                return render_template("result.html", table=result, image_path=image_path)
            except Exception as e:
                return f"Error: {str(e)}"
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
