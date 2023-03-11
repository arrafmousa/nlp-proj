# NLP PROJECT
# 212701239
# 207571258

import os
import nltk
import numpy as np
import torch
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def preprocess_train(filename):
    with open(filename, encoding="utf8") as f:
        lines = f.read()
    eng_entries = []
    ger_entries = []
    for entry in lines.split(2 * os.linesep):
        en_ger_couple = entry.split("English:\n")
        if len(en_ger_couple) <= 1:
            continue
        ger, eng = en_ger_couple[0].replace('German:\n', ''), en_ger_couple[1]
        eng_entries.append(eng)
        ger_entries.append(ger)
    return {"ger": ger_entries, "eng": eng_entries}


def train():

    # LOCAL FUNCTIONS
    def compute_metrics(eval_pred):
        predictions, labels_ = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels_ = np.where(labels_ != -100, labels_, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels_, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                         for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                          for label in decoded_labels]

        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                           for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_data_dict = preprocess_train(r'data/train.labeled')
    val_data_dict = preprocess_train(r'data/val.labeled')

    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_ds = Dataset.from_dict(train_data_dict)
    eval_ds = Dataset.from_dict(val_data_dict)

    print("Example of a training sample:")
    print(train_ds[0])
    print("Example of a testing sample:")
    print(eval_ds[0])

    # the following 2 hyper-parameters are task-specific
    max_source_length = 512
    max_target_length = 128

    # tokenize the training input
    input_sequences = train_ds["eng"]
    output_sequences = train_ds["ger"]

    # tokenize the validation input
    input_sequences_val = eval_ds["eng"]
    output_sequences_val = eval_ds["ger"]

    # prompting
    task_prefix = "translate German to English: "   # TODO check other prompting options

    # training set
    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(
        output_sequences,
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids

    # validation set
    encoding_val = tokenizer(
        [task_prefix + sequence for sequence in input_sequences_val],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids_val, attention_mask_val = encoding_val.input_ids, encoding_val.attention_mask

    # encode the targets
    target_encoding_val = tokenizer(
        output_sequences_val,
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    labels_val = target_encoding_val.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100
    labels_val[labels_val == tokenizer.pad_token_id] = -100

    # not sure of the format
    model_inputs = {"input": input_ids, "labels": labels}
    model_inputs_val = {"input": input_ids_val, "labels": labels_val}

    batch_size = 8
    epochs = 20
    model_file_name = "German_to_English_Translator"

    args = Seq2SeqTrainingArguments(
        model_file_name,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        # report_to="tensorboard"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    metric = load_metric("bleu")

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=model_inputs,
        eval_dataset=model_inputs_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

