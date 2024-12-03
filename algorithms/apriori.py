import pandas as pd
from itertools import combinations

def preprocess_data(data):
    """
    Tiền xử lý dữ liệu cho thuật toán Apriori.
    - Xử lý giá trị thiếu.
    - Mã hóa dữ liệu dạng chuỗi thành các giá trị nhị phân (One-Hot Encoding).
    """
    # Xử lý giá trị thiếu: thay thế giá trị thiếu bằng 0 hoặc loại bỏ dòng có giá trị thiếu
    data = data.fillna(0)

    # Mã hóa các giá trị chuỗi thành nhãn (sử dụng One-Hot Encoding)
    # Giả sử rằng mỗi cột là một sản phẩm, và mỗi dòng là một giao dịch
    return pd.get_dummies(data)

def calculate_support(data, itemset):
    """
    Tính hỗ trợ (support) của một itemset trong dữ liệu.
    """
    count = 0
    for _, transaction in data.iterrows():
        if all(item in transaction[transaction == 1].index for item in itemset):
            count += 1
    return count / len(data)

def generate_candidates(frequent_itemsets, k):
    """
    Sinh các tập hợp mục k từ các tập hợp mục k-1 phổ biến.
    """
    candidates = []
    frequent_items = list(frequent_itemsets.keys())
    for i in range(len(frequent_items)):
        for j in range(i + 1, len(frequent_items)):
            candidate = tuple(sorted(set(frequent_items[i]) | set(frequent_items[j])))
            if len(candidate) == k and candidate not in candidates:
                candidates.append(candidate)
    return candidates

def apriori(data, min_support):
    """
    Thuật toán Apriori để tìm các tập hợp mục phổ biến.
    """
    # Tìm các itemsets phổ biến đơn lẻ
    items = data.columns
    frequent_itemsets = {}
    k = 1
    candidates = [(item,) for item in items]
    
    while candidates:
        # Tính hỗ trợ cho từng candidate
        candidate_support = {}
        for candidate in candidates:
            support = calculate_support(data, candidate)
            if support >= min_support:
                candidate_support[candidate] = support
        
        # Lưu các tập hợp mục phổ biến k
        frequent_itemsets.update(candidate_support)
        k += 1
        
        # Sinh các candidates k+1
        candidates = generate_candidates(candidate_support, k)
    
    # Trả về các tập hợp mục phổ biến và hỗ trợ của chúng
    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence):
    """
    Sinh luật kết hợp từ các tập hợp mục phổ biến.
    """
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for k in range(1, len(itemset)):
                for antecedent in combinations(itemset, k):
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    antecedent_support = frequent_itemsets.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = support / antecedent_support
                        if confidence >= min_confidence:
                            rules.append({
                                'Antecedent': antecedent,
                                'Consequent': consequent,
                                'Confidence': confidence,
                                'Support': support
                            })
    return pd.DataFrame(rules)

def apriori_algorithm(data, min_support=0.3, min_confidence=0.7):
    """
    Chạy thuật toán Apriori để tìm các tập hợp mục phổ biến và sinh luật kết hợp.
    """
    try:
        # Tiền xử lý dữ liệu
        processed_data = preprocess_data(data)

        # Tìm tập hợp mục phổ biến
        frequent_itemsets = apriori(processed_data, min_support)
        if not frequent_itemsets:
            return "Không tìm thấy tập hợp mục phổ biến với min_support đã cho."

        # Sinh luật kết hợp
        rules = generate_rules(frequent_itemsets, min_confidence)
        if rules.empty:
            return "Không tìm thấy luật kết hợp nào với min_confidence đã cho."
        
        # Trả về luật kết hợp dưới dạng bảng HTML
        return rules.to_html()
    
    except Exception as e:
        return f"Lỗi trong thuật toán Apriori: {e}"
