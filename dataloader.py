import torch
from transformers import BertTokenizer

tokenizers = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#from torch.utils.data import Dataset


class IMDBDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = tokenizers

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())
        inputs = self.tokenizer.encode_plus(review, None,
                                            add_special_tokens=True,
                                            max_length=512,
                                            pad_to_max_length=True)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),

            "targets": torch.tensor(self.target[item], dtype=torch.float)
        }
