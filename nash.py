import re
import torch
import bertscore
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util, models
from bert_score import BERTScorer
from abc import ABC, abstractmethod

from utils import get_idf_dict
from num_similarity import NumericSimilarity

# this script is for the plug-in NASH for all transformer models
# implement the version for 1 pair, then extend to batch inferencing

# regular expression for number detection
core_num = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
acc_negative = f"\\({core_num}\\)"
standard_num = f"(?:-)?{core_num}"
NUMBER_PATTERN = f"(?<!\\d)(?:{acc_negative}|{standard_num})"


# --- Level 1: 定義 Text Scorer 的介面 ---
class BaseTextScorer(ABC):
    @abstractmethod
    def compute(self, text1, text2) -> float:
        pass

# --- Level 2: 實作具體的 Scorer ---

class CosineScorer(BaseTextScorer):
    """封裝 SentenceTransformer + Custom Pooling"""
    def __init__(self, model_name, pooling_strategy='mean', device='cpu'):
        self.device = device
        print(f"[CosineScorer] Loading {model_name} with {pooling_strategy.upper()} pooling...")
        
        try:
            # 這裡封裝了原本在 NASH __init__ 裡面的複雜邏輯
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
        # 統一介面：輸入字串，輸出 float
        emb1 = self.model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.model.encode(text2, convert_to_tensor=True, show_progress_bar=False)
        return util.cos_sim(emb1, emb2).item()


class BertScoreWrapper(BaseTextScorer):
    """封裝 BERTScore"""
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
        # BERTScore 需要 list 輸入
        cands = [text1]
        refs = [text2]
        P, R, F1 = self.model.score(cands, refs, verbose=False)
        return F1.item()


# --- Level 3: 清爽的 NASH 主程式 ---
class NASH:
    def __init__(
            self, 
            model_name='bert-base-uncased', 
            similarity_metric='cosine', # 'cosine' or 'bertscore'
            num_sim_scorer='paper',
            average_num_sim=False,
            pooling_strategy='mean',
            num_mask='[NUM]', 
            alignment_threshold=0.5,
            weighted_method='tf', 
            idf_dict=None,
            device=None,
        ):
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_mask = num_mask             
        self.number_pattern = NUMBER_PATTERN
        
        # 這裡還保留 tokenizer 是因為你後面算 Alpha 可能會用到
        # 但如果只有 Scorer 用得到，其實也可以移除
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # --- 漂亮的 Scorer 初始化 ---
        if similarity_metric == 'cosine':
            self.scorer = CosineScorer(model_name, pooling_strategy, self.device)
        elif similarity_metric == 'bertscore':
            self.scorer = BertScoreWrapper(model_name, self.device)
        else:
            raise ValueError(f"Unknown metric: {similarity_metric}")

        self.alignment_threshold = alignment_threshold
        self.weighted_method = weighted_method


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


    def text_similarity(self, mask_text1, mask_text2):
        return self.scorer.compute(mask_text1, mask_text2)


    def num_contextual_alignment(self, orig_text1, orig_text2, num_tokens1, num_tokens2):
        # get the token embeddings of original numbers
        for token in num_tokens1:
            # we need to get the embedding of the original number string from the original text
            inputs = self.tokenizer(orig_text1, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            # get by averaging the token embeddings in the span
            # e.g. I have $10,000. --> I have $ 10, 000. --> we need to get the embeddings of "10,000"
            span_start, span_end = token['span'] # character level id
            token_ids = self.tokenizer(orig_text1, return_offsets_mapping=True)['offset_mapping']
            token_indices = [i for i, (s, e) in enumerate(token_ids) if not (e <= span_start or s >= span_end)]
            token_embeddings = outputs.last_hidden_state[0, token_indices, :]
            token['embedding'] = torch.mean(token_embeddings, dim=0)

        for token in num_tokens2:
            inputs = self.tokenizer(orig_text2, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            span_start, span_end = token['span']
            token_ids = self.tokenizer(orig_text2, return_offsets_mapping=True)['offset_mapping']
            token_indices = [i for i, (s, e) in enumerate(token_ids) if not (e <= span_start or s >= span_end)]
            token_embeddings = outputs.last_hidden_state[0, token_indices, :]
            token['embedding'] = torch.mean(token_embeddings, dim=0)

        # compute alignment matrix
        alignment_matrix = torch.zeros((len(num_tokens1), len(num_tokens2)))
        for i, token1 in enumerate(num_tokens1):
            for j, token2 in enumerate(num_tokens2):
                sim = util.cos_sim(token1['embedding'], token2['embedding']).item()
                alignment_matrix[i, j] = sim

        # 2 directional: num in text1 to text2, and num in text2 to text1
        aligned_pairs = []
        for i, token1 in enumerate(num_tokens1):
            for j, token2 in enumerate(num_tokens2):
                if alignment_matrix[i, j] >= self.alignment_threshold:
                    aligned_pairs.append((token1, token2))

        return aligned_pairs


    def num_similarity(self, aligned_pairs):
        # Problem: 5 numbers, each difference is 20, total difference is 100 vs 1 number difference is 100
        # the similarity level should be closer
        if not aligned_pairs:
            return 0.0
        total_sim = 0.0
        for token1, token2 in aligned_pairs:
            val1 = token1['value']
            val2 = token2['value']
            diff = abs(val1 - val2)
            # pass


    def aggregate(self, mask_text1, mask_text2, text_sim, num_sim):
        # get alpha by idf or tf, we implement tf first
        if self.alpha is not None:
            score = alpha * text_sim + (1 - alpha) * num_sim

        elif self.weighted_method == 'tf':
            # calcuate the number of [NUM] (self.num_mask) in each text as (1 - alpha)
            tf_num = mask_text1.count(self.num_mask) + mask_text2.count(self.num_mask)
            total_tokens = len(self.tokenizer.tokenize(mask_text1)) + len(self.tokenizer.tokenize(mask_text2))
            alpha = 1 - (tf_num / total_tokens) if total_tokens > 0 else 1.0
            score = alpha * text_sim + (1 - alpha) * num_sim



    def predict_one_pair(self, text1, text2):
        # 這是之後要實作的入口
        pass






