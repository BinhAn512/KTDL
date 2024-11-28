import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def apriori_algorithm(data, min_support=0.5, metric="lift", min_threshold=1.0):
    """
    Chạy thuật toán Apriori để tìm các tập hợp mục phổ biến và tạo luật kết hợp.

    Args:
        data (pd.DataFrame): Dữ liệu đầu vào dạng One-Hot Encoding.
        min_support (float): Ngưỡng hỗ trợ tối thiểu.
        metric (str): Tiêu chí để tạo luật (ví dụ: 'lift', 'confidence').
        min_threshold (float): Ngưỡng tối thiểu cho tiêu chí metric.

    Returns:
        str: Luật kết hợp dưới dạng bảng HTML hoặc thông báo lỗi.
    """
    try:
        # Tìm các tập hợp mục phổ biến
        frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

        # Kiểm tra nếu không tìm thấy tập hợp mục
        if frequent_itemsets.empty:
            return "Không tìm thấy tập hợp mục nào với min_support đã cho."

        # Số lượng itemsets để truyền vào hàm association_rules
        num_itemsets = len(frequent_itemsets)

        # Tạo luật kết hợp
        rules = association_rules(
            frequent_itemsets, metric=metric, min_threshold=min_threshold, num_itemsets=num_itemsets
        )

        # Kiểm tra nếu không tìm thấy quy tắc kết hợp
        if rules.empty:
            return "Không tìm thấy luật kết hợp nào với các ngưỡng đã cho."

        # Trả về luật kết hợp dưới dạng HTML
        return rules.head().to_html()

    except Exception as e:
        return f"Lỗi trong thuật toán Apriori: {e}"


