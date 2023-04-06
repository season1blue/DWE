import argparse
from transformers.models.auto.modeling_auto import MODEL_MAPPING

MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)
def parse_arg():
    parser = argparse.ArgumentParser()

    # Path parameters

    parser.add_argument("--pretrain_model_path", default="../../data/pretrain_models", type=str)
    parser.add_argument("--cache_path", default="../../data/cache", type=str)
    parser.add_argument("--seg_model_path", default="../../data/pretrain_models/pointrend_resnet50.pkl", type=str)
    parser.add_argument("--model_name_or_path", default="../../data/pretrain_models/bert", type=str)
    parser.add_argument("--entity_filename", default="brief.json", type=str)
    parser.add_argument("--dataset", default="wiki", type=str)
    parser.add_argument("--img_path", default="../../data/ImgData", type=str)
    parser.add_argument("--log_path", default="../log.log", type=str)
    parser.add_argument("--gt_type", default="property", type=str)

    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")



    parser.add_argument(
        "--dir_prepro",
        default=None,
        type=str,
        required=True,
        help="The prepro data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--path_ans_list",
        default=None,
        type=str,
        help="Path to ordered answer list.",
    )
    parser.add_argument(
        "--path_candidates",
        default=None,
        type=str,
        help="Path to search results.",
    )
    parser.add_argument(
        "--path_neg_config",
        default="data/prepro_data/dm/neg_cache/neg.json",
        type=str,
        help="Path to neg.json",
    )
    parser.add_argument(
        "--dir_img_feat",
        default="data/prepro_data/nel",
        type=str,
        help="Path to image features."
    )
    parser.add_argument(
        "--dir_neg_feat",
        default="data/prepro_data/nel",
        type=str,
        help="Path to negative samples' features."
    )
    parser.add_argument(
        "--dir_eval",
        default=None,
        type=str,
        help="Path to eval model",
    )
    parser.add_argument(
        "--dir_output",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # model configs
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_sent_length",
        default=32,
        type=int,
        help="The maximum total input sequence length in nel model.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    # negative sample
    parser.add_argument(
        "--neg_sample_num",
        default=3,
        type=int,
        help="The num of negatives sample.",
    )

    parser.add_argument(
        "--img_len",
        default=196,
        type=int,
        help="The number of image regions.",
    )
    parser.add_argument(
        "--dropout",
        default=0.2,
        type=float,
        help="Dropout",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hidden_size", default=512, type=int, help="Hidden size."
    )
    parser.add_argument(
        "--ff_size", default=2048, type=int, help="Feed forward size."
    )
    parser.add_argument(
        "--text_feat_size", default=768, type=int, help="Text feature's size."
    )
    parser.add_argument(
        "--img_feat_size", default=2048, type=int, help="Image feature's size."  # 2048
    )
    parser.add_argument(
        "--nheaders", default=8, type=int, help="Num of attention headers."
    )
    parser.add_argument(
        "--num_attn_layers", default=1, type=int, help="Num of attention layers"
    )
    parser.add_argument(
        "--output_size", default=768, type=int, help="Output size."
    )
    parser.add_argument(
        "--feat_cate", default='wp', help="word (w), phrase(p) and sentence(s), default: wp"
    )
    parser.add_argument(
        "--rnn_layers", default=2, type=int, help="rnn_layers."
    )

    # loss
    parser.add_argument(
        "--loss_function", default="circle", type=str, help="Loss: triplet | circle"
    )

    parser.add_argument(
        "--loss_margin", default=0.25, type=float, help="margin of circle loss."
    )
    parser.add_argument(
        "--similarity", default='cos', type=str, help="Similarity"
    )
    # triplet loss
    parser.add_argument(
        "--loss_p", default=2, type=int, help="The norm degree for pairwise distance."
    )
    # circle loss
    parser.add_argument(
        "--loss_scale", default=32, type=int, help="Scale of circle loss."
    )

    # cross validation
    parser.add_argument("--do_cross", action="store_true", help="Whether to run cross validation.")
    parser.add_argument("--folds", default=5, type=int, help="Num of folds in cross validation.")
    parser.add_argument("--cross_arg", default=None, type=str, help="Arg to valid in cross validation.")
    parser.add_argument("--mode_pace", default='add', help="Pace mode of arg in validation: add|multiple")
    parser.add_argument("--pace", default=0.1, type=float, help="Pace of arg in cross validation.")
    parser.add_argument("--ub_arg", default=1, type=float, help="The upper bound of arg in cross validation.")

    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--do_cross_eval", action="store_true", help="Whether to run predictions on the cross set.")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")  # origin 50
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")  # origin 500
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--multi_gpus", action="store_true", help="Using multiple CUDAs when available")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--test_model_dir",
        default=None,
        type=str,
        help="The test model dir.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--gpu_id", type=int, default=0, help="id of gpu")
    parser.add_argument("--single_gpu", action="store_true", help="is single gpu?")

    return parser.parse_args()
