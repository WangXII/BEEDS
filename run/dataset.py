''' Custom Dataset for Distant SUpervision '''

import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DistantBertDataset(Dataset):

    def __init__(self, features, entity_blinding):
        self.features = features
        self.entity_blinding = entity_blinding

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = {
            "input_ids": torch.tensor(self.features[index].input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.features[index].input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(self.features[index].segment_ids, dtype=torch.long),
            "labels": torch.tensor(self.features[index].label_ids, dtype=torch.long),
            "subjects": torch.tensor(self.features[index].subjects, dtype=torch.long),
            "pubmed_ids": torch.tensor(self.features[index].pubmed_id, dtype=torch.long),
            "whitespace_bools": torch.tensor(self.features[index].whitespace_bools, dtype=torch.bool),
            "position_ids": torch.tensor(self.features[index].position_ids, dtype=torch.long),
            "question_ids": torch.tensor(self.features[index].question_id, dtype=torch.long),
            "subject_lengths": torch.tensor(self.features[index].subject_lengths, dtype=torch.long),
            "question_types": torch.tensor(self.features[index].question_type, dtype=torch.long),
            "debug_label_ids": torch.tensor(self.features[index].debug_label_ids, dtype=torch.long),
            "blinded_input_ids": torch.tensor(self.features[index].input_ids_blinded, dtype=torch.long),
        }

        if self.entity_blinding:
            sample["input_ids"] = torch.tensor(self.features[index].input_ids_blinded, dtype=torch.long)
            sample["blinded_input_ids"] = torch.tensor(self.features[index].input_ids, dtype=torch.long)

        return sample

    def truncate(self, size):
        # for feature in self.features:
        #     logger.info(torch.tensor(feature.input_ids, dtype=torch.long).size())
        #     logger.info(len(feature.input_ids))
        for i in range(len(self.features)):
            if len(self.features) > 0 and len(self.features[i].input_ids) > size:
                self.features[i].input_ids = self.features[i].input_ids[:size]
                self.features[i].input_mask = self.features[i].input_mask[:size]
                self.features[i].segment_ids = self.features[i].segment_ids[:size]
                self.features[i].label_ids = self.features[i].label_ids[:size]
                self.features[i].subjects = self.features[i].subjects[:size]
                self.features[i].pubmed_id = self.features[i].pubmed_id[:size]
                self.features[i].whitespace_bools = self.features[i].whitespace_bools[:size]
                self.features[i].position_ids = self.features[i].position_ids[:size]
                self.features[i].question_id = self.features[i].question_id[:size]
                self.features[i].subject_lengths = self.features[i].subject_lengths[:size]
                self.features[i].question_type = self.features[i].question_type[:size]
                self.features[i].debug_label_ids = self.features[i].debug_label_ids[:size]
                self.features[i].input_ids_blinded = self.features[i].input_ids_blinded[:size]
        # logger.warn("Feature lengths")
        # logger.warn(self.features[0].input_ids)
        # logger.warn(self.features[0].subject_lengths)
        # logger.warn(len(self.features[0].input_ids))
        # logger.warn(len(self.features[0].subject_lengths))
        # exit()
