import re
import torch
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util, models
from bert_score import BERTScorer
from sklearn.metrics.pairwise import cosine_similarity

# 假設 num_similarity.py 與此檔案在同一目錄
from num_similarity import NumericSimilarity

# --- Regex Definitions ---
core_num = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
acc_negative = f"\\({core_num}\\)"
standard_num = f"(?:-)?{core_num}"
NUMBER_PATTERN = f"(?<!\\d)(?:{acc_negative}|{standard_num})"


# --- Level 1: Text Scorer Interface ---
class BaseTextScorer(ABC):
    @abstractmethod
    def compute(self, text1, text2) -> float:
        pass


# --- Level 2: Concrete Scorers ---
class CosineScorer(BaseTextScorer):
    def __init__(self, model_name, pooling_strategy='mean', device='cpu'):
        self.device = device
        print(f"[CosineScorer] Loading {model_name} with {pooling_strategy.upper()} pooling...")
        try:
            word_embedding_model = models.Transformer(model_name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=(pooling_strategy == 'mean'),
                pooling_mode_cls_token=(pooling_strategy == 'cls'),
                pooling_mode_max_tokens=(pooling_strategy == 'max')
            )
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)
        except Exception as e:
            raise ValueError(f"Error initializing SentenceTransformer: {e}")

    def compute(self, text1, text2) -> float:
        emb1 = self.model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.model.encode(text2, convert_to_tensor=True, show_progress_bar=False)
        return util.cos_sim(emb1, emb2).item()


class BertScoreWrapper(BaseTextScorer):
    def __init__(self, model_name, device='cpu', batch_size=64):
        self.device = device
        print(f"[BertScoreWrapper] Loading {model_name}...")
        try:
            self.model = BERTScorer(
                model_type=model_name,
                device=self.device,
                batch_size=batch_size,
                lang="en",
                rescale_with_baseline=False
            )
        except Exception as e:
            raise ValueError(f"Error initializing BERTScorer: {e}")

    def compute(self, text1, text2) -> float:
        cands = [text1]
        refs = [text2]
        P, R, F1 = self.model.score(cands, refs, verbose=False)
        return F1.item()


# --- Level 3: NASH Main Class ---
class NASH:
    def __init__(
            self, 
            model_name='bert-base-uncased', 
            similarity_metric='cosine', # 'cosine' or 'bertscore'
            num_sim_method='paper',     # 'paper', 'range_neg1_pos1', 'log'
            pooling_strategy='mean',
            num_mask='[NUM]', 
            alignment_threshold=0.5,
            weighted_method='tf',       # 'tf' or 'idf'
            idf_dict=None,
            device=None,
        ):
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_mask = num_mask             
        self.number_pattern = NUMBER_PATTERN
        self.alignment_threshold = alignment_threshold
        self.weighted_method = weighted_method
        self.idf_dict = idf_dict
        self.num_sim_method = num_sim_method

        # 用於提取 Embedding (Contextual Alignment)
        # 注意：這裡使用 AutoModel 原生加載，因為我們需要 access hidden states
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 初始化 Text Scorer
        if similarity_metric == 'cosine':
            self.text_scorer = CosineScorer(model_name, pooling_strategy, self.device)
        elif similarity_metric == 'bertscore':
            self.text_scorer = BertScoreWrapper(model_name, self.device)
        else:
            raise ValueError(f"Unknown metric: {similarity_metric}")

    def modal_seperation(self, text):
        number_tokens = []
        
        for m in re.finditer(self.number_pattern, text):
            val_str = m.group()
            
            # --- 數值清理與轉換 ---
            clean_str = val_str.replace(',', '') 
            
            is_accounting_neg = False
            if clean_str.startswith('(') and clean_str.endswith(')'):
                is_accounting_neg = True
                clean_str = clean_str.replace('(', '').replace(')', '')
            
            try:
                val = float(clean_str)
                if is_accounting_neg:
                    val = -abs(val)
                
                number_tokens.append({
                    "value": val,
                    "str": val_str,
                    "span": m.span()
                })
            except ValueError:
                continue

        # --- Masking (由後往前) ---
        masked_text = text
        sorted_tokens = sorted(number_tokens, key=lambda x: x['span'][0], reverse=True)
        
        for token in sorted_tokens:
            start, end = token['span']
            masked_text = masked_text[:start] + self.num_mask + masked_text[end:]
            
        return number_tokens, masked_text

    def get_contextual_embeddings(self, text, num_tokens):
        """
        [優化] 一次 encode 整句，然後切片取出所有數字的 Embedding
        """
        if not num_tokens:
            return []

        # 1. Encode Full Text
        inputs = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs.pop('offset_mapping')[0].cpu().numpy()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [Batch, Seq, Dim] -> [Seq, Dim]
            embeddings = outputs.last_hidden_state[0]

        # 2. Slice Embeddings for each number
        token_embeddings_list = []
        for token in num_tokens:
            start_char, end_char = token['span']
            
            # 找出對應 character span 的 token indices
            # 條件: token 的 end > char_start 且 token 的 start < char_end
            indices = [
                i for i, (os, oe) in enumerate(offset_mapping) 
                if oe > os and max(start_char, os) < min(end_char, oe)
            ]
            
            if indices:
                # 取平均
                emb = torch.mean(embeddings[indices], dim=0).cpu().numpy()
                token_embeddings_list.append(emb)
            else:
                # Fallback: 如果 tokenizer 把這個數字切丟了 (極少見)，用整句平均代替
                # 避免 crash
                sent_emb = torch.mean(embeddings[1:-1], dim=0).cpu().numpy()
                token_embeddings_list.append(sent_emb)
                
        return token_embeddings_list

    def compute_numeric_score(self, nums1, nums2, embs1, embs2):
        """
        實作 Global Aggregation:
        計算 S1 -> S2 (Recall) 和 S2 -> S1 (Precision) 的雙向分數
        """
        # Edge Cases
        if not nums1 and not nums2: return 1.0
        if not nums1 or not nums2: return 0.0

        # 計算 Similarity Matrix
        sim_matrix = cosine_similarity(embs1, embs2)

        def _directional_score(source, target, matrix, axis):
            # axis=1: row-wise max (S1 -> S2)
            # axis=0: col-wise max (S2 -> S1)
            
            if axis == 0:
                sims = matrix.T # [len(s2), len(s1)]
            else:
                sims = matrix   # [len(s1), len(s2)]

            total_score = 0.0
            match_count = 0
            
            for i in range(len(source)):
                # 1. Best Match Alignment
                best_idx = np.argmax(sims[i])
                context_sim = sims[i][best_idx]
                
                # 2. Threshold Filtering
                if context_sim >= self.alignment_threshold:
                    val1 = source[i]['value']
                    val2 = target[best_idx]['value']
                    
                    # 3. Numeric Similarity Calculation (Delegate to helper)
                    if self.num_sim_method == 'paper':
                        sim_val = NumericSimilarity.paper_version(val1, val2)
                    elif self.num_sim_method == 'range_neg1_pos1':
                        sim_val = NumericSimilarity.range_neg1_pos1(val1, val2)
                    elif self.num_sim_method == 'log':
                        sim_val = NumericSimilarity.log_scale(val1, val2)
                    else:
                        sim_val = NumericSimilarity.paper_version(val1, val2)

                    total_score += sim_val
                    match_count += 1
            
            # Recall-style Penalty: 對齊到的總分 / 來源總數
            # 沒對齊到的視為 0 分
            return total_score / len(source)

        score_s1_to_s2 = _directional_score(nums1, nums2, sim_matrix, axis=1)
        score_s2_to_s1 = _directional_score(nums2, nums1, sim_matrix, axis=0)
        
        return (score_s1_to_s2 + score_s2_to_s1) / 2

    def aggregate(self, mask_text1, mask_text2, text_sim, num_sim):
        """
        計算 Alpha 並融合分數
        """
        alpha = 1.0 # Default text only

        if self.weighted_method == 'tf':
            # 簡單 TF 計算: 數字 Token 佔比
            # 我們計算 num_mask 出現的次數
            count1 = mask_text1.count(self.num_mask)
            count2 = mask_text2.count(self.num_mask)
            
            # 簡單估算總 Token 數 (用 split 粗估，或用 tokenizer)
            # 使用 tokenizer 比較準
            len1 = len(self.tokenizer.encode(mask_text1, add_special_tokens=False))
            len2 = len(self.tokenizer.encode(mask_text2, add_special_tokens=False))
            
            total_nums = count1 + count2
            total_tokens = len1 + len2
            
            if total_tokens > 0:
                ratio = total_nums / total_tokens
                # Alpha 是 Text 的權重，所以是 1 - ratio
                alpha = 1.0 - ratio
            else:
                alpha = 1.0
        
        elif self.weighted_method == 'idf' and self.idf_dict is not None:
            # 這裡需要實作 IDF 查找，稍微複雜一點，先保留介面
            # 如果你有 prepare_idf，可以在這裡 lookup
            # 暫時 fallback 到 0.5 或 TF
            pass

        return alpha * text_sim + (1 - alpha) * num_sim, alpha

    def predict_one_pair(self, text1, text2):
        """
        End-to-End Prediction
        """
        # 1. Separation
        nums1, mask1 = self.modal_seperation(text1)
        nums2, mask2 = self.modal_seperation(text2)
        
        # 2. Text Similarity
        s_text = self.text_scorer.compute(mask1, mask2)
        
        # 3. Numeric Embedding Extraction
        embs1 = self.get_contextual_embeddings(text1, nums1)
        embs2 = self.get_contextual_embeddings(text2, nums2)
        
        # 4. Numeric Similarity
        s_num = self.compute_numeric_score(nums1, nums2, embs1, embs2)
        
        # 5. Aggregation
        nash_score, alpha = self.aggregate(mask1, mask2, s_text, s_num)
        
        return {
            'score': nash_score,
            's_text': s_text,
            's_num': s_num,
            'alpha': alpha,
            'details': {
                'nums1': [n['str'] for n in nums1],
                'nums2': [n['str'] for n in nums2]
            }
        }

# --- Quick Test ---
if __name__ == "__main__":
    nash = NASH(similarity_metric='cosine', num_sim_method='paper', device='cpu')
    
    t1 = "EPS rose 10% to $11."
    t2 = "EPS was $10 in 2024, a 25% increase."
    
    res = nash.predict_one_pair(t1, t2)
    print("Test Pair Result:")
    print(res)
