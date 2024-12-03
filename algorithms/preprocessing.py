import pandas as pd

def preprocess_data(data):
    # Xóa giá trị null
    data = data.dropna()

    # Mã hóa dữ liệu dạng chuỗi
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].astype('category').cat.codes

    # Tính toán ma trận tương quan
    correlation_matrix = data.corr()

    # Chuyển cả bảng mô tả và ma trận tương quan sang HTML
    describe_html = data.describe().to_html()
    correlation_html = correlation_matrix.to_html()

    # Kết hợp cả hai bảng vào một chuỗi HTML
    combined_html = (
        "<h3>Bảng mô tả dữ liệu:</h3>" +
        describe_html +
        "<h3>Ma trận tương quan:</h3>" +
        correlation_html
    )

    return combined_html  # Trả về HTML hiển thị
