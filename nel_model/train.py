import os
import json
import logging
import argparse
import numpy as np
import torch
import random
import pickle
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm, trange
from os.path import join, exists
from glob import glob

from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import NELDataset, train_collate_fn, eval_collate_fn
from dataset import person_collate_eval, person_collate_train
from nel import NELModel
from metric_topk import cal_top_k, faiss_cal_topk
from time import time
from args import parse_arg
from dataset import load_and_cache_examples
import h5py

# 1、创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 2、创建一个handler，用于写入日志文件
fh = logging.FileHandler('../log.log')
fh.setLevel(logging.DEBUG)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 3、定义handler的输出格式（formatter）
formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt="%m/%d %H:%M:%S")
# 4、给handler添加formatter
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 5、给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

# 忽略not init权重的warning提示
from transformers import logging

logging.set_verbosity_error()

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, nel_model, answer_list, tokenizer, fold=""):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.dataset == "person" or "diverse":
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    else:
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=train_collate_fn)  # 1 iter

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)] + \
                   [p for n, p in nel_model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)] + \
                     [p for n, p in nel_model.named_parameters() if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [{"params": params_decay, "weight_decay": args.weight_decay},
                                    {"params": params_nodecay, "weight_decay": 0.0}, ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      no_deprecation_warning=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        logger.info("loading the existing model weights")
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        nel_model = torch.nn.DataParallel(nel_model)
        print("train", type(model), model.device_ids)

    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))

    global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
    best_result = [0, 0, 0, 0]

    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     try:
    #         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     except ValueError:
    #         global_step = 0
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    nel_model.zero_grad()

    set_seed(args)  # Added here for reproductibility

    epoch_start_time = time()
    step_start_time = None
    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        if epoch == epochs_trained:
            # logger.info(f"Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin.")
            print(f"Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin.")
        else:
            # logger.info(
            #     f"Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin ({(time() - epoch_start_time) / (epoch - epochs_trained):2f}s/epoch).")
            print(f"Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin ({(time() - epoch_start_time) / (epoch - epochs_trained):2f}s/epoch).")
        epoch_iterator = train_dataloader
        num_steps = len(train_dataloader)
        for step, batch in tqdm(enumerate(epoch_iterator), desc="Train", ncols=50, total=num_steps):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            nel_model.train()

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            # bert_inputs = {
            #     "input_ids": batch["input_ids"],
            #     "attention_mask": batch["input_mask"]
            # }
            # bert_out = model(**bert_inputs)

            if args.dataset == "person" or "diverse":
                nel_inputs = {
                    "model_type": args.dataset,
                    "total": batch["total_feature"].float(),
                    "text": batch["text_feature"].float(),
                    "mention": batch["mention_feature"].float(),
                    "pos_feats": batch["pos"],
                    "neg_feats": batch["neg"]
                }

            else:
                nel_inputs = {
                    "model_type": args.dataset,
                    "mention": batch["mention_feature"].float(),
                    "text": batch["text_feature"].float(),
                    "total": batch["total_feature"].float(),
                    "segement": batch["segement_feature"].float(),
                    "profile": batch["profile_feature"].float(),
                    "pos_feats": batch["pos"],
                    "neg_feats": batch["neg"]
                }

            outputs = nel_model(**nel_inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(nel_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                nel_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    if step_start_time is None:
                        step_start_time = time()
                        print()
                        logger.info(
                            f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch + 1}: {step + 1}/{num_steps}")
                    else:
                        log_tim = (time() - step_start_time)
                        print()
                        # logger.info(
                        #     f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch + 1}: {step + 1}/{num_steps} ({log_tim:.2f}s/50step)")
                        logger.info(
                            f"epoch {epoch + 1}, loss: {(tr_loss - logging_loss) / args.logging_steps}")
                        step_start_time = time()
                    logging_loss = tr_loss

                # save model if args.save_steps>0
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # logger.info(f"\n******** Evaluation ********", )
                        results, _ = evaluate(args, model, nel_model, answer_list, tokenizer, mode=f"dev{fold}")[:2]
                        show_result = list(results.values())
                        if show_result[0] > best_result[0]:
                            best_result = show_result
                            best_result.append(epoch)
                        logger.info(
                            "### EVAL RESULT: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} at {4}".format(show_result[0],
                                                                                                show_result[1],
                                                                                                show_result[2],
                                                                                                show_result[3], epoch))
                        logger.info(
                            "### BEST RESULT: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} at {4}".format(best_result[0],
                                                                                                best_result[1],
                                                                                                best_result[2],
                                                                                                best_result[3],
                                                                                                best_result[4]))

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

            if 0 < args.max_steps < global_step:
                break

        if 0 < args.max_steps < global_step:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, nel_model, answer_list, tokenizer, mode, prefix=""):
    time_eval_beg = time()

    eval_dataset, guks = load_and_cache_examples(args, tokenizer, answer_list, mode=mode, dataset=args.dataset,
                                                 logger=logger)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    if args.dataset in ["person", "diverse"]:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
    else:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=eval_collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1 and type(model) not in {"torch.nn.parallel.data_parallel.DataParallel"}:
        model = torch.nn.DataParallel(model)
        nel_model = torch.nn.DataParallel(nel_model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    nel_model.eval()

    all_ranks = []
    time_eval_rcd = time()
    nsteps = len(eval_dataloader)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader), desc='Eval', ncols=50, total=nsteps):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            if args.dataset in ["person", "diverse"]:
                nel_inputs = {
                    "model_type": args.dataset,
                    "total": batch["total_feature"].float(),
                    "text": batch["text_feature"].float(),
                    "mention": batch["mention_feature"].float(),
                    "pos_feats": batch["pos"],
                    "neg_feats": batch["neg"]
                }

            else:
                nel_inputs = {
                    "model_type": args.dataset,
                    "mention": batch["mention_feature"].float(),
                    "text": batch["text_feature"].float(),
                    "total": batch["total_feature"].float(),
                    "segement": batch["segement_feature"].float(),
                    "profile": batch["profile_feature"].float(),
                    "pos_feats": batch["pos"],
                    "neg_feats": batch["neg"]
                }

            outputs = nel_model(**nel_inputs)
            tmp_eval_loss, query = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss

            pos_feat_trans = nel_model.trans(batch["pos"])
            neg_feat_trans = nel_model.trans(batch["search_res"])

            rank_list, sim_p, sim_n = cal_top_k(args, query, pos_feat_trans, neg_feat_trans)

            all_ranks.extend(rank_list)

            nb_eval_steps += 1

            if (i + 1) % 100 == 0:
                print(f"{mode}: {i + 1}/{nsteps}, loss: {tmp_eval_loss}, {time() - time_eval_rcd:.2f}s/100steps")
                time_eval_rcd = time()
    eval_loss = eval_loss.item() / nb_eval_steps
    all_ranks = np.array(all_ranks)
    results = {
        # "mean_rank": sum(all_ranks) / len(all_ranks) + 1,
        "top1": int(sum(all_ranks <= 1)) / len(eval_dataset),
        "top5": int(sum(all_ranks <= 5)) / len(eval_dataset),
        "top10": int(sum(all_ranks <= 10)) / len(eval_dataset),
        "top20": int(sum(all_ranks <= 20)) / len(eval_dataset),
        # "top50": int(sum(all_ranks <= 50))/len(eval_dataset),
        # "all": len(all_ranks),
        # "loss": float(eval_loss)
    }

    # logger.info("***** Eval results %s *****", prefix)
    logger.info(f"Eval loss: {eval_loss}, Eval time: {time() - time_eval_beg:2f}")

    return results, eval_loss, all_ranks


def recover_nel_args(args, args_his):
    args.neg_sample_num = args_his.neg_sample_num
    args.dropout = args_his.dropout

    args.hidden_size = args_his.hidden_size
    args.ff_size = args_his.ff_size
    args.nheaders = args_his.nheaders
    args.num_attn_layers = args_his.num_attn_layers
    args.output_size = args_his.output_size
    args.text_feat_size = args_his.text_feat_size
    args.img_feat_size = args_his.img_feat_size
    args.feat_cate = args_his.feat_cate
    args.rnn_layers = args_his.rnn_layers

    args.loss_scale = args_his.loss_scale
    args.loss_margin = args_his.loss_margin
    args.loss_function = args_his.loss_function
    args.similarity = args_his.similarity


def load_entity(args):
    gt_name = "gt_feats_{}.h5".format(args.gt_type)
    entity_feat = h5py.File(join(args.dir_neg_feat, gt_name), 'r')
    entity_features = entity_feat.get("features")

    entity_features = torch.tensor(entity_features)
    # entity_features = np.array(entity_features)
    return entity_features


def main():
    args = parse_arg()
    args.n_gpu = 0

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.warning(
        "========Process rank: %s, device: %s，16-bits training: %s========", args.local_rank, args.device, args.fp16)

    # Set seed
    set_seed(args)

    answer_list = json.load(open(args.path_ans_list))
    args.model_type = args.model_type.lower()

    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}

    if args.do_cross and (args.do_train or args.do_eval or args.do_predict):
        raise ValueError("You shouldn't do eval or predict or train When do cross")

    # Training
    if args.do_train:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
            **tokenizer_args,
        )
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # Initialize the nel model
        nel_model = NELModel(args)
        path_nel_state = join(args.model_name_or_path, "nel_model.pkl")
        if exists(path_nel_state):
            logger.info(f"Load nel model state dict from {path_nel_state}")
            nel_model.load_state_dict(torch.load(path_nel_state, map_location=lambda storage, loc: storage))

            path_args_history = join(args.model_name_or_path, "training_args.bin")
            args_his = torch.load(path_args_history)
            recover_nel_args(args, args_his)
        else:
            for p in nel_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # logger.info("Device: %s", args.device)
        model.to(args.device)
        nel_model.to(args.device)

        train_dataset, _ = load_and_cache_examples(args, tokenizer, answer_list, mode="train", dataset=args.dataset,
                                                   logger=logger)
        global_step, tr_loss = train(args, train_dataset, model, nel_model, answer_list, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
