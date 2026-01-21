import numpy as np
import math

class NumericSimilarity:
    """
    提供多種計算兩個數字 v1, v2 相似度的數學機制。
    """

    @staticmethod
    def paper_version(v1, v2):
        """
        1. Follow Paper 定義 (NASH 原版)
        公式: 1 / (1 + abs(v1-v2) / (1 + 0.5*(|v1| + |v2|)))
        特色: 
          - 範圍 [0, 1]
          - 考慮相對誤差 (Relative Error)
          - 1 是常數平滑項，避免分母為 0
        """
        diff = abs(v1 - v2)
        # Normalization factor: 隨著數值越大，容忍的絕對誤差越大
        norm = 1 + 0.5 * (abs(v1) + abs(v2))
        
        # 轉成相似度
        return 1 / (1 + (diff / norm))

    @staticmethod
    def range_neg1_pos1(v1, v2):
        """
        2. 控制在 [-1, 1] 之間
        邏輯: 將原本 [0, 1] 的分數線性映射到 [-1, 1]。
        公式: Score_new = 2 * Score_paper - 1
        特色:
          - 1.0 代表完全相同
          - 0.0 代表誤差等於 Norm (中立)
          - -1.0 代表誤差極大 (完全相反/無關)
        """
        base_score = NumericSimilarity.paper_version(v1, v2)
        return 2 * base_score - 1

    @staticmethod
    def log_scale(v1, v2):
        """
        3. 有取 Log (處理數量級差異)
        邏輯: 使用 Signed Log 轉換，專攻「數量級」比較。
        公式: f(x) = sign(x) * log(1 + |x|)
             Score = 1 / (1 + |f(v1) - f(v2)|)
        特色:
          - 對長尾分佈 (Long-tail) 數據較好 (如 10億 vs 10億零100)
          - 10 vs 100 的距離 == 100 vs 1000 的距離
        """
        def signed_log(x):
            # 使用 log(1+|x|) 避免 log(0) 且保持連續性
            return np.sign(x) * np.log1p(abs(x))
        
        log_v1 = signed_log(v1)
        log_v2 = signed_log(v2)
        
        dist = abs(log_v1 - log_v2)
        return 1 / (1 + dist)

# --- 測試案例 (驗證 Edge Cases) ---
if __name__ == "__main__":
    test_cases = [
        (10, 10, "完全相同"),
        (10, 11, "微小誤差"),
        (10, 20, "兩倍差異"),
        (10, 100, "數量級差異 (10x)"),
        (10, 1000, "數量級差異 (100x)"),
        (10, -10, "正負相反"),
        (0.05, 0.50, "5% vs 50% (小數)"),
        (1000000, 1000100, "大數微小差異"),
        (0, 0, "零值比較"),
    ]

    print(f"{'Case':<20} | {'Paper [0,1]':<12} | {'Range [-1,1]':<12} | {'Log Scale':<12}")
    print("-" * 65)

    for v1, v2, desc in test_cases:
        s1 = NumericSimilarity.paper_version(v1, v2)
        s2 = NumericSimilarity.range_neg1_pos1(v1, v2)
        s3 = NumericSimilarity.log_scale(v1, v2)
        
        print(f"{v1} vs {v2:<10} | {s1:.4f}       | {s2:.4f}       | {s3:.4f}")
