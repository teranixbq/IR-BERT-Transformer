"""
BERT Cross-Encoder for Neural Re-Ranking
Dosen: Zico Pratama Putra
Kelompok: [14250028 - Hanief Fathul Bahri Ahmad],[14250029 - Irfan Wibowo],[14250027 - Muhammad Arief Nadhofa]
"""

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class ManualExtractiveQAPipeline:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_seq_len: int = 384,
        doc_stride: int = 128,
        n_best_size: int = 20,
    ):
        self.device = device
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.n_best_size = n_best_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        question: str,
        context: str,
        handle_impossible_answer: bool = True,
        max_answer_len: int = 50,
        **kwargs,
    ):
        question = str(question).strip()
        context = str(context).strip()

        if not question or not context:
            return {"answer": "", "score": 0.0, "start": None, "end": None}

        encoded = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_len,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)

        sequence_ids_per_feature = [
            encoded.sequence_ids(i) for i in range(encoded["input_ids"].shape[0])
        ]

        model_inputs = encoded.to(self.device)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        start_logits = outputs.start_logits.detach().cpu()
        end_logits = outputs.end_logits.detach().cpu()
        offset_mapping = offset_mapping.cpu().tolist()

        best_answer = {"answer": "", "score": 0.0, "start": None, "end": None}

        for feature_idx in range(start_logits.shape[0]):
            start_probs = torch.softmax(start_logits[feature_idx], dim=-1)
            end_probs = torch.softmax(end_logits[feature_idx], dim=-1)

            start_top = torch.topk(
                start_probs, k=min(self.n_best_size, start_probs.shape[0])
            ).indices.tolist()

            end_top = torch.topk(
                end_probs, k=min(self.n_best_size, end_probs.shape[0])
            ).indices.tolist()

            sequence_ids = sequence_ids_per_feature[feature_idx]
            offsets = offset_mapping[feature_idx]

            for start_idx in start_top:
                for end_idx in end_top:
                    if sequence_ids[start_idx] != 1 or sequence_ids[end_idx] != 1:
                        continue

                    if end_idx < start_idx:
                        continue

                    if (end_idx - start_idx + 1) > max_answer_len:
                        continue

                    char_start, _ = offsets[start_idx]
                    _, char_end = offsets[end_idx]

                    if char_end <= char_start:
                        continue

                    answer_text = context[char_start:char_end].strip()
                    if not answer_text:
                        continue

                    score = float(start_probs[start_idx] * end_probs[end_idx])

                    if score > best_answer["score"]:
                        best_answer = {
                            "answer": answer_text,
                            "score": score,
                            "start": int(char_start),
                            "end": int(char_end),
                        }

        return best_answer
