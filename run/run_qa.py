# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
    Runs biomedical event extraction as multi-turn question answering.
'''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import glob
import logging
import numpy as np
import torch
import wandb
import math

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW, WEIGHTS_NAME, AutoConfig

from configs import OWL_STATEMENTS, TQDM_DISABLE, ModelHelper
from configs.parser import parse_arguments
from data_processing.biopax_to_retrieval import IndraDataLoader
from data_processing.datatypes import LABELS, Question, QUESTION_TYPES, COMPLEX_TYPES_MAPPING, PAD_TOKEN_LABEL_ID
from metrics.eval import get_average_precision, visualize
from metrics.sequence_labeling import precision_score, recall_score, f1_score, stats  # , classification_report
from run.bert_model import ModelForDistantSupervision
from run.roberta_model import ModelForDistantSupervision2
from run.utils_io import DataBuilderLoader, save_model, set_seed
from run.utils_run import update_with_nn_output, initialize_loaders, update_metadata, extract_from_nn_output, collate_fn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def check_datastring(datastring, mode):
    if ("train" in datastring) and ("dev" not in datastring) and ("test" not in datastring):
        return "train"
    elif ("dev" in datastring) and ("train" not in datastring) and ("test" not in datastring):
        return "eval"
    elif ("test" in datastring) and ("train" not in datastring) and ("dev" not in datastring):
        return "test"
    else:
        raise ValueError(
            "Malformed String. String {}_data must contain exactly one of the three strings (train, dev, test).".format(mode))


class Excecutor:
    ''' Main entry point for Biomedical Event Extraction with Distant Supervision '''

    def __init__(self):
        self.parser = parse_arguments()
        self.args = self.parser.parse_args()
        self.pad_token_label_id = PAD_TOKEN_LABEL_ID
        self.labels = LABELS
        self.num_labels = len(self.labels)
        self.device = torch.device("cpu")
        self.n_gpu = 0
        self.question_types = []
        self.model_helper = None
        self.model = None
        if "roberta" in self.args.model_name_or_path:
            self.distant_supervision_model = ModelForDistantSupervision2
        else:
            self.distant_supervision_model = ModelForDistantSupervision

    def run(self):
        if self.args.wandb:
            wandb.init(project="masterarbeit")
            wandb.config.update(self.args)

        # Check valid train_data, dev_data and test_data names
        check_datastring(self.args.train_data, "train")
        check_datastring(self.args.dev_data, "dev")
        check_datastring(self.args.test_data, "test")

        # Setup CUDA, GPU & distributed training
        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        # Set seed
        set_seed(self.args, self.n_gpu)

        # Setup logging
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       self.args.local_rank, self.device, self.n_gpu, bool(self.args.local_rank != -1), self.args.fp16)

        # Load pretrained model and tokenizer (in model_helper)
        if self.args.do_train or self.args.do_eval or self.args.do_predict:
            if self.args.local_rank not in [-1, 0]:
                # Make sure only the first process in distributed training will download model & vocab
                torch.distributed.barrier()

            config = AutoConfig.from_pretrained(self.args.model_name_or_path, hidden_dropout_prob=self.args.dropout,
                                                num_labels=self.num_labels, gradient_checkpointing=False)
            self.model = self.distant_supervision_model(
                config=config, multi_instance_bool=self.args.multi_instance_learning,
                multi_instance_bool_neg=self.args.multi_instance_learning_neg,
                crf_bool=self.args.crf, model_name=self.args.model_name_or_path)

            if self.args.local_rank == 0:
                # Make sure only the first process in distributed training will download model & vocab
                torch.distributed.barrier()

            self.model.to(self.device)

            logger.info("Training/evaluation parameters %s", self.args)
            if self.args.wandb:
                wandb.watch(self.model, log="all")

        self.model_helper = ModelHelper(self.args.model_name_or_path)
        if self.args.train_data.startswith("basic"):
            self.question_types = [Question.PHOSPHORYLATION_CAUSE]
        elif self.args.train_data.startswith("complex"):
            self.question_types = [Question.PHOSPHORYLATION_COMPLEXCAUSE]
        elif len(self.args.question_types) > 0:
            self.question_types = [Question(int(number)) for number in self.args.question_types]
        else:
            self.question_types = [Question(number) for number in QUESTION_TYPES]

        data_loader = DatasetLoader(self.args, self.model_helper, self.labels, self.pad_token_label_id)
        trainer = Trainer(self.args, self.model_helper, self.labels, self.pad_token_label_id)
        evaluator = Evaluator(self.args, self.model_helper, self.labels, self.pad_token_label_id)
        predictor = Evaluator(self.args, self.model_helper, self.labels, self.pad_token_label_id)

        # Anomaly detection
        # torch.autograd.set_detect_anomaly(True)

        # Build datasets
        if self.args.do_build_data:
            data_loader.load_from_indra_and_bionlp(self.question_types, self.args.train_data, datasplit="train")
            data_loader.load_from_indra_and_bionlp(self.question_types, self.args.dev_data, datasplit="eval")
            data_loader.load_from_indra_and_bionlp(self.question_types, self.args.test_data, datasplit="test")

        # Training
        if self.args.do_train:
            train_dataset, train_direct_dataset, _, _ = data_loader.load_from_indra_and_bionlp(self.question_types, self.args.train_data, datasplit="train")
            train_dataset = ConcatDataset(train_dataset)
            train_direct_dataset = ConcatDataset(train_direct_dataset)
            # exit()
            eval_datasplit = check_datastring(self.args.dev_data, "dev")
            trainer.run(self.model, train_dataset, train_direct_dataset, self.device,
                        self.n_gpu, data_loader, evaluator, self.question_types, eval_datasplit)

        # Evaluation
        if self.args.do_eval and self.args.local_rank in [-1, 0]:
            eval_datasplit = check_datastring(self.args.dev_data, "dev")
            if self.args.cache_predictions in [0, 1, 2]:
                eval_dataset, eval_direct_dataset, distant_encoder, direct_encoder = data_loader.load_from_indra_and_bionlp(
                    self.question_types, self.args.dev_data, datasplit=eval_datasplit)
            else:
                eval_dataset = eval_direct_dataset = distant_encoder = direct_encoder = [[] for i in range(25)]
            number_questions = len(eval_dataset)

            # Load pretrained tokenizer (in model_helper) and model
            self.model_helper = ModelHelper(self.args.model_name_or_path, self.args.output_dir)
            checkpoints = [self.args.output_dir]
            if self.args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(
                    self.args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                config = None
                self.model = self.distant_supervision_model(
                    config=config, model_name=checkpoint, multi_instance_bool=self.args.multi_instance_learning,
                    multi_instance_bool_neg=self.args.multi_instance_learning_neg, crf_bool=self.args.crf)
                self.model.to(self.device)
                logger.info("Start evaluation for directly supervised data ...")
                losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, _ = \
                    evaluator.run(self.model, eval_direct_dataset, self.device, self.n_gpu, eval_datasplit,
                                  checkpoint_prefix=global_step, label_encoders=direct_encoder, cache_predictions=0)
                evaluator.evaluate_nn(nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, eval_datasplit, losses,
                                      supervision="Direct", checkpoint_prefix=global_step,
                                      visualize_bool=self.args.visualize_preds)
                logger.info("Start evaluation for distantly supervised data ...")
                losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events = \
                    evaluator.run(self.model, eval_dataset, self.device, self.n_gpu, eval_datasplit,
                                  checkpoint_prefix=global_step, label_encoders=distant_encoder,
                                  cache_predictions=self.args.cache_predictions)
                evaluator.evaluate(nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, eval_datasplit, losses,
                                   checkpoint_prefix=global_step, visualize_bool=self.args.visualize_preds)
                if self.args.multiturn:
                    logger.info("Start multiturn evaluation...")
                    del eval_dataset
                    _, eval_dataset, label_encoders = data_loader.load_from_prior_answers(
                        self.question_types, indra_events, self.args.dev_data, datasplit=eval_datasplit, cache=False)
                    losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events = \
                        evaluator.run(self.model, eval_dataset, self.device, self.n_gpu, eval_datasplit,
                                      checkpoint_prefix=global_step, label_encoders=label_encoders,
                                      cache_predictions=1, offset=number_questions)
                    evaluator.evaluate(nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions,
                                       eval_datasplit, losses, checkpoint_prefix=global_step, visualize_bool=self.args.visualize_preds,
                                       wandb_log=False)

        # Prediction
        if self.args.do_predict and self.args.local_rank in [-1, 0]:
            eval_datasplit = "custom"
            checkpoint = self.args.output_dir
            custom_dataset, label_encoders = data_loader.load_custom_queries()
            self.model_helper = ModelHelper(self.args.model_name_or_path, self.args.output_dir)
            config = None
            self.model = self.distant_supervision_model(
                config=config, model_name=checkpoint, multi_instance_bool=self.args.multi_instance_learning,
                multi_instance_bool_neg=self.args.multi_instance_learning_neg, crf_bool=self.args.crf)
            self.model.to(self.device)
            for i, question_dataset in enumerate(custom_dataset):
                output_cache_file = self.args.cache_dir + "/processed_output_custom_" + str(i) + ".npz"
                _, _, _, groundtruth, predictions, indra_events = predictor.run_question_type(
                    self.model, question_dataset, self.device, self.n_gpu, eval_datasplit, i, label_encoder=label_encoders[i],
                    output_cache_file=output_cache_file, cache_predictions=self.args.cache_predictions)
                if len(indra_events) > 0:
                    logger.info("Visualization of predictions")
                    visualize(groundtruth, predictions)
                else:
                    logger.info("No predictions/Retrieval empty")

        if self.args.wandb:
            wandb.save("model.h1")


class DatasetLoader:
    ''' Class for loading data '''

    def __init__(self, args, model_helper, labels, pad_token_label_id):
        self.dataset = None
        self.dataset_array = []
        self.direct_dataset_array = []
        self.label_encoders = []
        self.args = args
        self.model_helper = model_helper
        self.labels = labels
        self.pad_token_label_id = pad_token_label_id
        self.loader = DataBuilderLoader(self.args, self.model_helper, self.labels, self.pad_token_label_id)

    def load_from_indra_and_bionlp(self, question_types, dataset_name, datasplit, cache=True):
        ''' Load distantly supervised datasets from INDRA and directly supervised data from BioNLP '''
        self.dataset_array = []
        self.distant_encoder = []

        # Distantly supervised INDRA/PID dataset
        for i, question_type in enumerate(question_types):
            _, event_dict = IndraDataLoader.get_dataset(mode=datasplit, question_type=question_type, biopax_model_str=OWL_STATEMENTS)
            if event_dict is not None:
                dataset, distant_subject_encoder = self.loader.load_and_cache_examples(dataset_name, event_dict, question_type, cache=cache)
                # logger.info(len(dataset))
                # logger.info(dataset[0])
                self.dataset_array.append(dataset)
                self.distant_encoder.append(distant_subject_encoder)

        # Directly supervised BioNLP dataset
        self.direct_dataset_array = []
        self.direct_encoder = []
        for i, question_type in enumerate(question_types):
            dataset, direct_subject_encoder = self.loader.load_and_cache_direct_examples(
                self.args.direct_data, question_type, datasplit, cache=cache, add_negative_examples=self.args.negative_examples)
            # logger.info(len(dataset))
            # logger.info(dataset[0])
            if len(dataset) > 0:
                logger.debug("Build direct data")
                logger.debug(question_type)
                logger.debug(len(dataset))
                self.direct_dataset_array.append(dataset)
                self.direct_encoder.append(direct_subject_encoder)

        return self.dataset_array, self.direct_dataset_array, self.distant_encoder, self.direct_encoder

    def load_from_prior_answers(self, question_types, prior_events, dataset_name, datasplit=None, true_subset=True, cache=False):
        ''' Build datasets from prior events for multi-turn questions. '''
        self.dataset_array = []
        self.label_encoders = []

        for i, question_type in enumerate(question_types):
            if question_type.value in COMPLEX_TYPES_MAPPING.keys():
                # logger.info(question_type.value)
                # logger.info(prior_events.keys())
                prior_question_type = Question(COMPLEX_TYPES_MAPPING[question_type.value])
                if prior_question_type.name in prior_events:
                    # logger.info(prior_question_type.name)
                    prior_event_dict = prior_events[prior_question_type.name]
                    logger.info(prior_question_type.name)
                    logger.info(len(prior_event_dict))
                    if true_subset:
                        _, event_dict = IndraDataLoader.get_dataset(mode=datasplit, question_type=question_type, biopax_model_str=OWL_STATEMENTS)
                        prior_event_keys = event_dict.keys() & prior_event_dict.keys()
                        prior_event_dict = {k: prior_event_dict[k] for k in prior_event_keys}
                    logger.info(len(prior_event_dict))
                    if len(prior_event_dict) > 0:
                        dataset, distant_subject_encoder = self.loader.load_and_cache_examples(
                            "multiturn_" + dataset_name, prior_event_dict, question_type, cache=cache, predict_bool=True)
                        # logger.info(len(dataset))
                        # logger.info(dataset[0])
                        self.dataset_array.append(dataset)
                        self.label_encoders.append(distant_subject_encoder)

        self.dataset = ConcatDataset(self.dataset_array)
        return self.dataset, self.dataset_array, self.label_encoders

    def load_custom_queries(self, dataset_name="custom"):
        ''' Used to build custom queries in prediction mode. '''
        self.dataset_array = []
        self.label_encoders = []
        question_type_infos = [(e.name, e.value) for e in Question]
        question_numbers = [e.value for e in Question]

        while True:
            input_string = input(("Enter the question type number to build custom queries for. \n"
                                  "Enter 'help' to give an overview of available question numbers. \n"
                                  "Enter 'x' to finish the input and start predicting: \n"))
            if input_string == "help":
                print(question_type_infos)
                continue
            elif input_string == "x":
                break
            elif not input_string.isdigit():
                print("Malformed input string. Question type has to be a number!")
                continue
            question_number = int(input_string)
            if question_number > max(question_numbers) or question_number < min(question_numbers):
                print("Malformed input string. Question number is out of expected number range!")
                continue
            question_type = Question(question_number)
            event_dict = IndraDataLoader.get_custom_dataset(question_type=question_type)
            logger.info(event_dict)
            if event_dict is not None:
                dataset, distant_subject_encoder = self.loader.load_and_cache_examples(
                    dataset_name, event_dict, question_type, False, True)
                # logger.info(len(dataset))
                # logger.info(dataset[0])
                self.dataset_array.append(dataset)
                self.label_encoders.append(distant_subject_encoder)

        # self.dataset = ConcatDataset(self.dataset_array)
        # return self.dataset, self.dataset_array, self.label_encoders
        return self.dataset_array, self.label_encoders


class ModelRunner:
    ''' Base class for Trainer, Evaluator and Predictor. '''

    def __init__(self, args, model_helper, labels, pad_token_label_id):
        self.args = args
        self.model_helper = model_helper
        self.labels = labels
        self.pad_token_label_id = pad_token_label_id


class Trainer(ModelRunner):

    def __init__(self, args, model_helper, labels, pad_token_label_id):
        super(Trainer, self).__init__(args, model_helper, labels, pad_token_label_id)

    def run(self, model, dataset, direct_dataset, device, n_gpu, data_loader, evaluator, question_types, eval_datasplit):
        """ Model trainer """

        # for feature in dataset:
        #     logger.info("Length {}".format(feature["input_ids"].size()))

        if os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(self.args.output_dir))

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        train_dataloader, train_direct_dataloader, scheduler = initialize_loaders(self.args, dataset, direct_dataset, optimizer)

        if self.args.fp16:
            scaler = torch.cuda.amp.GradScaler()
        if "berta" in self.args.model_name_or_path:
            if n_gpu > 1:
                model.roberta = torch.nn.DataParallel(model.roberta)
            if self.args.local_rank != -1:
                model.roberta = torch.nn.parallel.DistributedDataParallel(
                    model.roberta, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                    find_unused_parameters=True)
        else:
            if n_gpu > 1:
                model.bert = torch.nn.DataParallel(model.bert)
            if self.args.local_rank != -1:
                model.bert = torch.nn.parallel.DistributedDataParallel(
                    model.bert, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                    find_unused_parameters=True)
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        global_step = 0
        tr_loss = 0.0
        for param in model.parameters():
            param.grad = None
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0])
        set_seed(self.args, n_gpu)  # Added here for reproductibility (even between python 2 and 3)

        # If using auxillary directly supervised data, draw random permutation of either dataset
        # 0 for distantly supervised, 1 for directly supervised data
        if self.args.use_directly_supervised_data and self.args.use_distantly_supervised_data:
            dataset_choice = [0] * len(train_dataloader) + [1] * len(train_direct_dataloader)
        elif self.args.use_directly_supervised_data:
            dataset_choice = [1] * len(train_direct_dataloader)
        else:
            dataset_choice = [0] * len(train_dataloader)
        np.random.shuffle(dataset_choice)

        # "pubmed_ids": torch.tensor(self.features[index].pubmed_id, dtype=torch.long),
        # "whitespace_bools": torch.tensor(self.features[index].whitespace_bools, dtype=torch.bool),
        # "position_ids": torch.tensor(self.features[index].position_ids, dtype=torch.long),
        # "question_ids": torch.tensor(self.features[index].question_id, dtype=torch.long),
        # "subject_lengths": torch.tensor(self.features[index].subject_lengths, dtype=torch.long),
        # "question_types": torch.tensor(self.features[index].question_type, dtype=torch.long),
        # "debug_label_ids": torch.tensor(self.features[index].debug_label_ids, dtype=torch.long),
        model.train()

        for iteration in train_iterator:
            logger.info("  Iteration = %d", iteration)
            epoch_iterator = tqdm(dataset_choice, desc="Iteration", disable=self.args.local_rank not in [-1, 0] or TQDM_DISABLE)
            distant_data_iterator = iter(train_dataloader)
            direct_data_iterator = iter(train_direct_dataloader)
            logger.debug(dataset_choice)
            logger.info("Length distantly supervised data = {}".format(len(distant_data_iterator)))
            logger.info("Length directly supervised data = {}".format(len(direct_data_iterator)))
            for step, dataset_index in enumerate(epoch_iterator):
                logger.debug("Step {} with {}".format(step, dataset_index))
                # Get current batch
                if dataset_index == 0:
                    batch = next(distant_data_iterator)
                else:  # dataset_index == 1
                    batch = next(direct_data_iterator)
                # Prepare Batch
                pad_sequences = torch.any(batch["attention_mask"], 1)
                pad_seq_id = (pad_sequences.long() == 1).sum()
                batch = {k: v[:pad_seq_id].to(device) for i, (k, v) in enumerate(batch.items()) if i <= 3}
                # logger.info(len(batch["input_ids"]))

                # Forward Pass
                if self.args.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                # exit()
                loss, logit_hist, logits = outputs[:3]  # model outputs are always tuple in pytorch-transformers (see doc)
                logit_hist = logit_hist.detach().cpu().numpy()
                logit_hist = np.nan_to_num(logit_hist)

                if loss.item() in [float("inf"), float("-inf")] or math.isnan(loss.item()):
                    logger.warn("Encountered invalid loss. Continue with next batch.")
                    continue

                # If directly supervised data, scale loss accordingly
                if dataset_index == 1:
                    loss = self.args.direct_weight * loss

                # Backward Pass
                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update gradients
                tr_loss += loss.item()
                # logger.info(loss.item())

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            scaler.unscale_(optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.args.wandb:
                        wandb.log({"average loss": tr_loss / global_step})
                        wandb.log({"gradient norm": total_norm})
                        wandb.log({"logits histogram": wandb.Histogram(np_histogram=np.histogram(
                            logit_hist, range=(np.nanmin(logit_hist), np.nanmax(logit_hist))))})
                        wandb.log({"logits max": np.max(logit_hist)})
                        wandb.log({"logits min": np.min(logit_hist)})

            # Save after an epoch
            if self.args.local_rank in [-1, 0] and self.args.save_steps > 0:
                output_dir = os.path.join(self.args.output_dir, "checkpoint-iteration-{}".format(iteration))
                save_model(self.args, output_dir, model)

            # Evaluate eval set in prediction settings
            if self.args.local_rank in [-1, 0] and self.args.evaluate_during_training:
                eval_dataset, eval_direct_dataset, distant_encoder, direct_encoder = data_loader.load_from_indra_and_bionlp(
                    question_types, self.args.dev_data, datasplit=eval_datasplit)
                number_questions = len(eval_dataset)
                logger.info("Start evaluation for directly supervised data ...")
                losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events = \
                    evaluator.run(model, eval_direct_dataset, device, n_gpu, eval_datasplit, train_model=True,
                                  checkpoint_prefix=global_step, label_encoders=direct_encoder, cache_predictions=0)
                evaluator.evaluate_nn(nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, eval_datasplit, losses,
                                      supervision="Direct", checkpoint_prefix=global_step)
                logger.info("Start evaluation for distantly supervised data ...")
                losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events = \
                    evaluator.run(model, eval_dataset, device, n_gpu, eval_datasplit, train_model=True,
                                  checkpoint_prefix=global_step, label_encoders=distant_encoder,
                                  cache_predictions=self.args.cache_predictions)
                evaluator.evaluate(nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, eval_datasplit, losses, checkpoint_prefix=global_step)
                if self.args.multiturn:
                    logger.info("Start multiturn evaluation ...")
                    del eval_dataset
                    _, eval_dataset, label_encoders = data_loader.load_from_prior_answers(question_types, indra_events, datasplit=eval_datasplit)
                    losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events = \
                        evaluator.run(model, eval_dataset, device, n_gpu, eval_datasplit, train_model=True,
                                      checkpoint_prefix=global_step, label_encoders=label_encoders,
                                      cache_predictions=self.args.cache_predictions, offset=number_questions)
                    evaluator.evaluate(nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, eval_datasplit, losses, checkpoint_prefix=global_step,
                                       wandb_log=False)
                logger.info("Finished evaluating")
                model.train()

            # Synchronize
            if self.args.local_rank != -1:
                torch.distributed.barrier()

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        if self.args.local_rank in [-1, 0]:
            save_model(self.args, self.args.output_dir, model)
            self.model_helper.tokenizer.save_pretrained(self.args.output_dir)


class Evaluator(ModelRunner):

    def __init__(self, args, model_helper, labels, pad_token_label_id):
        super(Evaluator, self).__init__(args, model_helper, labels, pad_token_label_id)

    def run(self, model, dataset, device, n_gpu, datasplit, train_model=False, checkpoint_prefix="",
            label_encoders=None, output_cache_file=None, cache_predictions=0, offset=0):
        ''' Evaluate a dataset. '''
        events_dict = {}
        kb_groundtruth = []
        kb_predictions = []
        nn_groundtruth = []
        nn_predictions = []
        losses = []
        for i, question_dataset in enumerate(dataset):
            question_number = i + offset
            loss, nn_ground, nn_pred, kb_ground, kb_pred, indra_events = \
                self.run_question_type(model, question_dataset, device, n_gpu, datasplit, question_number, train_model, checkpoint_prefix,
                                       label_encoder=label_encoders[i], output_cache_file=output_cache_file,
                                       cache_predictions=cache_predictions)
            events_dict = {**events_dict, **indra_events}
            kb_groundtruth.append(kb_ground)
            kb_predictions.append(kb_pred)
            nn_groundtruth.append(nn_ground)
            nn_predictions.append(nn_pred)
            losses.append(loss)
        return losses, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, events_dict

    def run_question_type(self, model, dataset, device, n_gpu, datasplit, question_type, train_model=False, checkpoint_prefix="",
                          label_encoder=None, output_cache_file=None, cache_predictions=0):
        """ Evaluate one question type of a whole dataset. """

        logger.info("Evaluating")
        eval_loss = 0

        if cache_predictions in [0, 1]:
            # multi-gpu evaluate, if using train_model, model is already parallelized!
            # No distributed evaluations available (and not really needed)
            if "berta" in self.args.model_name_or_path:
                if n_gpu > 1 and not train_model and not hasattr(model.roberta, "module"):
                    model.roberta = torch.nn.DataParallel(model.roberta)
            else:
                if n_gpu > 1 and not train_model and not hasattr(model.bert, "module"):
                    model.bert = torch.nn.DataParallel(model.bert)
            model.eval()

            # logger.info(list(range(len(eval_dataset) - 7, len(eval_dataset))))
            # eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(len(eval_dataset) - 71, len(eval_dataset))))
            # eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(10)))

            eval_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

            logger.info("***** Running evaluation %s *****", checkpoint_prefix)
            logger.info("  Question %s! ", question_type)
            logger.info("  Num examples = %d", len(dataset))
            # logger.info("  Num examples = %d", len(eval_dataloader))

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = out_label_ids = out_token_ids = subjects = subject_lengths = pubmed_ids = whitespace_bools = position_ids = question_ids = None
            question_types_ex = out_debug_label_ids = out_blinded_token_ids = None
            all_preds = attention_masks = None

            if len(dataset) == 0:
                return eval_loss, {}, {}, {}, {}, {}

            description = "Evaluating" if self.args.do_predict is False else "Predicting"
            if TQDM_DISABLE:
                logger.info(description)
            for batch in tqdm(eval_dataloader, desc=description, disable=TQDM_DISABLE):
                pad_sequences = torch.any(batch["attention_mask"], 1)
                pad_seq_id = (pad_sequences.long() == 1).sum()

                batch_cpu = [v[:pad_seq_id] for i, (k, v) in enumerate(batch.items()) if i > 3]
                # No need to push everything to GPU
                batch = {k: v[:pad_seq_id].to(device) for i, (k, v) in enumerate(batch.items()) if i <= 3}

                with torch.no_grad():
                    outputs = model(**batch)
                    loss, logit_hist, logits, sequence_prediction = outputs
                    eval_loss += loss.item()

                nb_eval_steps += 1

                preds, all_preds, out_label_ids, out_token_ids, attention_masks = update_with_nn_output(
                    batch, sequence_prediction, logits, preds, all_preds, out_label_ids, out_token_ids, attention_masks, self.args.crf)
                pubmed_ids, subjects, whitespace_bools, position_ids, question_ids, subject_lengths, question_types_ex, out_debug_label_ids, out_blinded_token_ids = \
                    update_metadata(
                        batch_cpu[1], batch_cpu[0], batch_cpu[2], batch_cpu[3], batch_cpu[4], batch_cpu[5], batch_cpu[6], batch_cpu[7], batch_cpu[8], pubmed_ids,
                        subjects, whitespace_bools, position_ids, question_ids, subject_lengths, question_types_ex, out_debug_label_ids, out_blinded_token_ids,
                        label_encoder)

            eval_loss = eval_loss / nb_eval_steps

            if cache_predictions == 1:  # and args.local_rank in [-1, 0]:
                filename = self.args.cache_dir + "/nn_output_question_" + str(question_type) + self.args.predictions_suffix + ".npz"
                np.savez(filename, preds, out_label_ids, out_token_ids, subjects, subject_lengths, pubmed_ids, whitespace_bools, position_ids,
                         question_ids, all_preds, attention_masks, question_types_ex, out_debug_label_ids, out_blinded_token_ids)

        elif cache_predictions == 2:
            filename = self.args.cache_dir + "/nn_output_question_" + str(question_type) + self.args.predictions_suffix + ".npz"
            logger.info("Load cached output from {}".format(filename))
            if not os.path.exists(filename):
                return -1, [], [], {}, {}, {}
            npzcache = np.load(filename, allow_pickle=True)
            preds, out_label_ids, out_token_ids, subjects, subject_lengths, pubmed_ids, whitespace_bools, position_ids, question_ids, \
                all_preds, attention_masks, question_types_ex, out_debug_label_ids, out_blinded_token_ids = [npzcache[array] for array in npzcache.files]
            eval_loss = 0

        nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events = \
            self._transform_nn_to_kb_output(
                model, train_model, datasplit, question_type,
                preds, out_label_ids, out_token_ids, subjects, subject_lengths, pubmed_ids, whitespace_bools, position_ids,
                question_ids, all_preds, attention_masks, question_types_ex, out_debug_label_ids, out_blinded_token_ids, output_cache_file,
                cache_predictions)
        # logger.info(len(indra_events))

        return eval_loss, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, indra_events

    def _transform_nn_to_kb_output(
            self, model, train_model, datasplit, question_type,
            preds, out_label_ids, out_token_ids, subjects, subject_lengths, pubmed_ids, whitespace_bools, position_ids,
            question_ids, all_preds, attention_masks, question_types_ex, out_debug_label_ids, out_blinded_token_ids, output_cache_file=None,
            cache_predictions=0):
        ''' Transform Neural Network Output to Knowledge Base Outputs '''

        if output_cache_file is None:
            filename = self.args.cache_dir + "/processed_output_question_" + str(question_type) + self.args.predictions_suffix + ".npz"
        else:
            filename = output_cache_file

        if cache_predictions in [0, 1]:
            logger.debug(len(preds))
            logger.debug(len(preds[0]))
            # preds = preds.reshape(out_label_ids.shape[0], out_label_ids.shape[1], preds.shape[2])
            # preds = np.argmax(preds, axis=2)
            # preds = np.array(preds)
            # Pad sequences smaller than max_seq_length
            try:
                preds = np.array([xi + [0] * (self.args.max_seq_length - len(xi)) for xi in preds])
            except ValueError:
                preds = np.array([xi + [0] * (self.args.max_seq_length - len(xi)) for xi in preds.tolist()])
            out_label_ids = out_label_ids.reshape(preds.shape[0], preds.shape[1])

            # logger.info(out_label_ids)
            # logger.info(preds)

            logger.debug("Shape Groundtruth")
            logger.debug(out_label_ids.shape)
            logger.debug("Shape Predictions")
            logger.debug(preds.shape)
            logger.debug("Shape Predictions (Logits)")
            logger.debug(all_preds.shape)

            logger.debug("Labels")
            logger.debug(self.labels)

            if self.args.entity_blinding:
                temp_token_ids = out_token_ids
                out_token_ids = out_blinded_token_ids
                out_blinded_token_ids = temp_token_ids

            # index are on token basis, not character
            groundtruth, predictions, out_label_list, preds_list, indra_preds, debug_scores = extract_from_nn_output(
                self.labels, out_label_ids, preds, all_preds, out_token_ids, attention_masks, whitespace_bools, position_ids,
                self.model_helper.tokenizer, pubmed_ids, subjects, question_ids, subject_lengths, question_types_ex, out_debug_label_ids, model, self.args)
            # logger.info(out_label_ids.shape)
            # logger.info(len(out_label_list))
            # logger.info(pubmed_ids)
            # logger.info(indra_preds)
            # logger.info(indra_preds.items()[0])
            # Plot bag scores for main answer entity of each question in wandb histograms
            # if self.args.wandb and self.args.eval_debug and not train_model:
            #     for debug_score_example in debug_scores:
            #         wandb.log({"Bag_Scores_{}".format(datasplit): wandb.Histogram(np.array(debug_score_example, dtype=object))})

            if cache_predictions == 1 or output_cache_file is not None:
                np.savez(filename, groundtruth, predictions, np.array(out_label_list, dtype=object), np.array(preds_list, dtype=object), indra_preds)

        elif cache_predictions == 2:
            npzcache = np.load(filename, allow_pickle=True)
            groundtruth, predictions, out_label_list, preds_list, indra_preds = [npzcache[array] for array in npzcache.files]
            # Loading dicts with np.load()
            groundtruth = groundtruth.item()
            predictions = predictions.item()
            indra_preds = indra_preds.item()

        return out_label_list, preds_list, groundtruth, predictions, indra_preds

    def evaluate(self, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, datasplit, losses, supervision="Distant", checkpoint_prefix="",
                 visualize_bool=False, wandb_log=True):
        ''' Evaluate the results and calculate the performance metrics '''
        results = {
            "loss": np.mean(losses),
            "nn_precision": precision_score(nn_groundtruth[0], nn_predictions[0]),
            "nn_recall": recall_score(nn_groundtruth[0], nn_predictions[0]),
            "nn_f1": f1_score(nn_groundtruth[0], nn_predictions[0]),
            "kb_average_precision":
                [question_result for question_result in get_average_precision(kb_groundtruth, kb_predictions, mode=datasplit,
                 use_db_ids=True, visualize_bool=visualize_bool)]
        }

        logger.info("***** Eval results %s *****", checkpoint_prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        if self.args.wandb and wandb_log:
            for key in sorted(results.keys()):
                # if key == "nn_f1":
                #     wandb.log({"NN_F1_{}_{}".format(datasplit, supervision): results[key]})
                # elif key == "nn_recall":
                #     wandb.log({"NN_Recall_{}_{}".format(datasplit, supervision): results[key]})
                # elif key == "nn_precision":
                #     wandb.log({"NN_Precision_{}_{}".format(datasplit, supervision): results[key]})
                # elif key == "loss":
                if key == "loss":
                    wandb.log({"Eval_Loss_{}_{}".format(datasplit, supervision): results[key]})
                elif key == "kb_average_precision":
                    for i, question_type_results in enumerate(results[key]):
                        qa_type = question_type_results[5]
                        if qa_type in [" All"]:
                            wandb.log({"KB_Average_Precision_{}_{}_{}".format(datasplit, qa_type, supervision): question_type_results[0]})
                            wandb.log({"KB_Recall_{}_{}_{}".format(datasplit, qa_type, supervision): question_type_results[1]})
                            wandb.log({"KB_Precision_{}_{}_{}".format(datasplit, qa_type, supervision): question_type_results[2]})
                            wandb.log({"KB_F1_{}_{}_{}".format(datasplit, qa_type, supervision): question_type_results[3]})
                            wandb.log({"KB_Number_Preds_{}_{}_{}".format(datasplit, qa_type, supervision): question_type_results[4]})

    def evaluate_nn(self, nn_groundtruth, nn_predictions, kb_groundtruth, kb_predictions, datasplit, losses, supervision="Distant", checkpoint_prefix="",
                    visualize_bool=False):
        ''' Evaluate the results and calculate the performance metrics. Only evaluate NN part and not knowledge base part.
            Used for evaluation of directly supervised data.
        '''

        # Visualize
        if visualize_bool:
            visualize_bionlp = get_average_precision(kb_groundtruth, kb_predictions, mode=datasplit, use_db_ids=True,
                                                     visualize_bool=visualize_bool, only_visualize=True)
            next(visualize_bionlp)

        results = []
        results_all = [0, 0, 0]
        for i in range(len(nn_groundtruth)):
            question_type_string = list(kb_groundtruth[i].values())[0][0][0][0]
            results.append(("Precision {}".format(question_type_string), precision_score(nn_groundtruth[i], nn_predictions[i])))
            results.append(("Recall {}".format(question_type_string), recall_score(nn_groundtruth[i], nn_predictions[i])))
            results.append(("F1 {}".format(question_type_string), f1_score(nn_groundtruth[i], nn_predictions[i])))
            nb_events = stats(nn_groundtruth[i], nn_predictions[i])
            results_all[0] += nb_events[0]
            results_all[1] += nb_events[1]
            results_all[2] += nb_events[2]
            results.append(("Stats (Correct, Predicted, Truth) {}".format(question_type_string), nb_events))
        precision_all = results_all[0] / results_all[1]
        recall_all = results_all[0] / results_all[2]
        f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)
        results.append(("Precision All", precision_all))
        results.append(("Recall All", recall_all))
        results.append(("F1 All", f1_all))
        results.append(("Stats (Correct, Predicted, Truth) All", results_all))

        logger.info("***** Eval results %s *****", checkpoint_prefix)
        logger.info("Bio NLP (directly supervised) data")
        for key in results:
            logger.info("  %s = %s", key[0], str(key[1]))
        logger.info("*****")

        if self.args.wandb:
            for key in results:
                if key[0] == "F1 All":
                    wandb.log({"BioNLP_F1_{}_{}".format(datasplit, supervision): key[1]})
                elif key[0] == "Recall All":
                    wandb.log({"BioNLP_Recall_{}_{}".format(datasplit, supervision): key[1]})
                elif key[0] == "Precision All":
                    wandb.log({"BioNLP_Precision_{}_{}".format(datasplit, supervision): key[1]})


if __name__ == "__main__":
    excecutor = Excecutor()
    excecutor.run()
