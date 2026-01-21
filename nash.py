import re
import torch
import bertscore
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util, models
from bert_score import BERTScorer

from utils import get_idf_dict

# this script is for the plug-in NASH for all transformer models
# implement the version for 1 pair, then extend to batch inferencing

# regular expression for number detection
core_num = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
acc_negative = f"\\({core_num}\\)"
standard_num = f"(?:-)?{core_num}"
number_pattern = f"(?<!\\d)(?:{acc_negative}|{standard_num})"

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
            pooling_strategy='mean',
            num_mask='[NUM]', 
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
        return self.text_sim_metric.compute_similarity(mask_text1, mask_text2)



    def


