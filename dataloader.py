import torch
from transformers import BertTokenizer
#123
tokenizers = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#from torch.utils.data import Dataset
def convert(lst):
    # print(lst)
    return ([i for item in lst for i in item.split()])

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
        # review = review.split()
        # review = convert(review)
        # if len(review) > 512:
        #     review = review[:128] + review[-382:]
        # review = ' '.join([str(elem) for elem in review])

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
