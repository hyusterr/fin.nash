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

class NASH:
    def __init__(
            self, 
            model_name='bert-base-uncased', 
            similarity_metric='cosine',
            pooling_strategy='mean',
            num_mask='[NUM]', # can try [MASK]?
            weighted_method='tf', # can try 'idf'
            idf_dict=None,
            alpha=None,
            device=None,
        ):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if similarity_metric not in ['cosine', 'euclidean', 'manhattan', 'bertscore']:
            raise ValueError("similarity_metric must be one of 'cosine', 'euclidean', or 'manhattan'")

        if similarity_metric == 'cosine':
            try:
                word_embedding_model = models.Transformer(model_name, device=self.device)
                pooling_model = models.Pooling(
                        word_embedding_model.get_word_embedding_dimension(), 
                        pooling_strategy=pooling_strategy
                )
                self.text_sim_metric = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)
            except Exception as e:
                raise ValueError(f"Error initializing SentenceTransformer: {e}")

        elif similarity_metric == 'bertscore':
            try:
                self.text_sim_metric = BERTScorer(
                    model_name, 
                    device=self.device
                    rescale_with_baseline=False
                )
        else:
            self.text_sim_metric = similarity_metric



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


