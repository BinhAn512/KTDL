from mlxtend.frequent_patterns import apriori, association_rules

def apriori_algorithm(data):
    try:
        # Chạy thuật toán Apriori để tìm các itemsets thường xuyên
        frequent_itemsets = apriori(data, min_support=0.5, use_colnames=True)

        # Tạo các quy tắc kết hợp từ các itemsets thường xuyên
        # Không cần tham số num_itemsets, chỉ cần metric và min_threshold
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        # Trả về các quy tắc kết hợp
        return rules.head().to_html()  # Hiển thị các quy tắc kết hợp đầu tiên dưới dạng bảng HTML
    except Exception as e:
        return f"Lỗi trong thuật toán Apriori: {e}"
