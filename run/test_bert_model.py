''' Unit testing for bert_model.py '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import unittest
import torch

from transformers import BertConfig

from run.bert_model import BertForDistantSupervision


class TestBertForDistantSupervision(unittest.TestCase):

    def setUp(self):
        self.config = BertConfig(num_labels=3)
        # torch.cuda.set_device(0)
        self.device = torch.device("cuda")
        self.model = BertForDistantSupervision(config=self.config)
        # print(torch.cuda.current_device())
        self.model.to(self.device)
        self.sequence_length = 16
        self.batch_size = 4
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, 100, (self.batch_size, self.sequence_length), device=self.device)
        self.labels = torch.randint(2, (self.batch_size, self.sequence_length), device=self.device)
        self.token_type_ids = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.int64, device=self.device)
        self.attention_mask = torch.ones((self.batch_size, self.sequence_length), dtype=torch.int64, device=self.device)

    def testForward(self):
        output = self.model.forward(input_ids=self.input_ids, attention_mask=self.attention_mask,
                                    token_type_ids=self.token_type_ids, labels=self.labels)
        print(output[0])
        self.assertEqual(output[0].size(), torch.Size([1]), "Wrong output shape for the loss function!")


if __name__ == '__main__':
    unittest.main()
