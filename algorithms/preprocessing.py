import pandas as pd

def preprocess_data(data):
    # Xóa giá trị null
    data = data.dropna()

    # Mã hóa dữ liệu dạng chuỗi
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].astype('category').cat.codes

    return data.describe().to_html()  # Trả về báo cáo HTML

