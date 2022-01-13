import os, re
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from KoBERT_NER.utils import init_logger, load_tokenizer, get_labels, xlsx2label

# 인풋 파일 읽기

def read_input_file(input_file_path):
    lines = []
    with open (input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip ()
            line = line.replace ('{S}', '')
            line = line.replace ('\n', '')
            words = line.split ()
            lines.append (words)

    return lines

# args 가져오기
def get_args(model_dir):
    return torch.load (model_dir + '/training_args.bin')


# 모델 가져오기
def load_model(model_dir, device):
    # Check whether model exists
    if not os.path.exists (model_dir):
        raise Exception ("모델이 없습니다! 훈련부터 하세요!")

    try:
        model = AutoModelForTokenClassification.from_pretrained (
            model_dir)  # Config will be automatically loaded from model_dir
        model.to (device)
        model.eval ()
        print ("***** Model Loaded *****")
    except:
        raise Exception ("모델 파일이 부족합니다...")

    return model

# 인풋 파일 텐서로 변환
def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize (word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend (word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend ([0] + [pad_token_label_id] * (len (word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len (tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len (tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids (tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len (input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len (input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append (input_ids)
        all_attention_mask.append (attention_mask)
        all_token_type_ids.append (token_type_ids)
        all_slot_label_mask.append (slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor (all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor (all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor (all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor (all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset (all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def predict(pred_model, pred_label, pred_input, pred_batch_size, label_sheet_name=None):
    # load model and args
    if torch.cuda.is_available ():
        device = torch.device ("cuda")
        print ('There are %d GPU(s) available.' % torch.cuda.device_count ())
        print ('We will use the GPU:', torch.cuda.get_device_name (0))
    else:
        device = torch.device ("cpu")
        print ('No GPU available, using the CPU instead.')

    model = load_model (pred_model, device)
    args = get_args (pred_model)
    print(args)
    
    ## label 정보가 DefaultDic으로 들어왔을 때 처리
    if pred_label == os.getcwd () + r'/DefaultDic.xlsx':
        label_lst = xlsx2label(pred_label, label_sheet_name)
    else:
        label_path = pred_label
        label_lst = get_labels (label_path, is_predict=True)
    

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_input)
    dataset = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_batch_size)

    all_slot_label_mask = None
    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])
    
    # Write to output file
    with open("%s_pred_out.txt" % pred_input[:-4], "w", encoding="utf-8") as f:
        for words, preds in zip(lines, preds_list):
            line = ""
            n = 1
            for word, pred in zip(words, preds):
                if pred == 'O':
                    line = line + word + " "
                
                # 기존 LGG 태그 우선시
                elif re.search(r"<.+?>[^\n\<]+</.+?>", word):
                    line = line + word + ' '

                else:
                    # pred = pred.replace('__','=').replace('_','-')
                    pred = pred.replace('_','=')                        # 0113 replace 이유 FEA_PO-B
                    if pred[-2:] == '-B':
                        try:
                            if preds[n][-2:] == '-I':
                                line = line + "<{}>{} ".format(pred[:-2], word)
                            else:
                                line = line + "<{}>{}</{}> ".format(pred[:-2], word, pred[:-2].split('=')[0])
                        except IndexError:
                            line = line + "<{}>{}</{}> ".format(pred[:-2], word, pred[:-2].split('=')[0])
                    else:
                        try:
                            if preds[n][-2:] == '-I':
                                line = line + word + " "
                            elif preds[n-2][-2:] == 'O':
                                line = line + word + " "
                            else:
                                line = line + "{}</{}> ".format(word, pred[:-2].replace("-", "=").split('=')[0])
                        except IndexError:
                            line = line + "{}</{}> ".format(word, pred[:-2].replace("-", "=").split('=')[0])
                n += 1
            
            # max_seq_length 때문에 잘리는 부분 복구

            if len(words) != len(preds):
                for i in range(len(preds), len(words)):
                    line = line + words[i] + " "

            f.write("{}\n".format(line.strip()))
    

    with open("%s_pred_out.txt" % pred_input[:-4], "r", encoding='utf-8') as R:
        return R.read(), preds_list

    # logger.info("Prediction Done!")



