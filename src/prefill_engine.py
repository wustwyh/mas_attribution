
import gc
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


class PrefillEngine:

    def __init__(self, model_path: str, device: str = None, attn_layer_ratio: float = 0.2):

        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.attn_layer_ratio = attn_layer_ratio

        print(f"[PrefillEngine] Loading tokenizer from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[PrefillEngine] Loading model to {self.device} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True,
            device_map="auto" if self.device == 'cuda' else None,
            attn_implementation="eager",
        )
        if self.device == 'cpu':
            self.model = self.model.to('cpu')
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.attn_layers = list(range(
            int(self.num_layers * (1 - attn_layer_ratio)),
            self.num_layers
        ))
        print(f"[PrefillEngine] Layers: {self.num_layers}, extract attention from: {self.attn_layers}")

    def prefill(self, text: str, step_token_ranges: List[Tuple[int, int]],
                system_prompt: str = "") -> Dict[str, np.ndarray]:

        if system_prompt:
            full_text = system_prompt + "\n\n" + text
            sys_ids = self.tokenizer.encode(system_prompt, add_special_tokens=True)
            offset = len(sys_ids)
        else:
            full_text = text
            sys_ids = self.tokenizer.encode("", add_special_tokens=True)
            offset = len(sys_ids)

        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        seq_len = input_ids.shape[1]

        adjusted_ranges = []
        for s, e in step_token_ranges:
            adj_s = s + offset
            adj_e = e + offset
            adj_s = max(0, min(adj_s, seq_len))
            adj_e = max(0, min(adj_e, seq_len))
            adjusted_ranges.append((adj_s, adj_e))

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        captured_attns = []

        def _capture_attn(module, input, output):

            attn_weights = output[1] if isinstance(output, tuple) else getattr(output, 'attn_weights', None)
            if attn_weights is not None:
                captured_attns.append(attn_weights.detach())

        hooks = [
            self.model.model.layers[i].self_attn.register_forward_hook(_capture_attn)
            for i in self.attn_layers
            if i < self.num_layers
        ]

        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=False, use_cache=False)
                logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        log_probs = torch.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, 1:]
        log_probs = log_probs[:, :-1, :]
        token_nll = -log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll.squeeze(0).cpu().float().numpy()
        del logits, log_probs

        num_steps = len(adjusted_ranges)
        step_nll = np.zeros(num_steps, dtype=np.float32)
        for idx, (s, e) in enumerate(adjusted_ranges):
            nll_start = max(0, s - 1)
            nll_end = max(0, e - 1)
            if nll_end > nll_start:
                step_nll[idx] = float(token_nll[nll_start:nll_end].mean())
            else:
                step_nll[idx] = 0.0

        attn_matrix = None
        if captured_attns:
            for layer_attn in captured_attns:
                layer_np = layer_attn.mean(dim=1).squeeze(0).cpu().float().numpy()
                attn_matrix = layer_np if attn_matrix is None else attn_matrix + layer_np
            attn_matrix /= len(captured_attns)
        else:
            attn_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
        del outputs, captured_attns

        step_attention = np.zeros((num_steps, num_steps), dtype=np.float32)
        for i in range(num_steps):
            si, ei = adjusted_ranges[i]
            if si >= ei:
                continue
            for j in range(i):
                sj, ej = adjusted_ranges[j]
                if sj >= ej:
                    continue
                block = attn_matrix[si:ei, sj:ej]
                if block.size > 0:
                    step_attention[i, j] = float(block.mean())

        del input_ids, token_nll
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return {
            'step_nll': step_nll,
            'step_attention': step_attention,
        }

    def estimate_max_tokens_per_step(self, num_steps: int, total_budget: int = 6000,
                                     reserve: int = 200) -> int:

        return max(64, (total_budget - reserve) // num_steps)
