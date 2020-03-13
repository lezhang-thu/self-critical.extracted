import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/coco.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                        help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/cocotalk_box',
                        help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--cached_tokens', type=str, default='')

    # Model settings
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_per_img', type=int, default=5)

    # Sample related
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_method = greedy, indicates number of beams in beam search.')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Maximum length during sampling')

    # Optimization: for the Language Model
    parser.add_argument('--learning_rate', type=float, default=4e-4)

    parser.add_argument('--val_images_use', type=int, default=3200)
    parser.add_argument('--save_checkpoint_every', type=int, default=2500)

    parser.add_argument('--losses_log_every', type=int, default=25)

    # misc
    parser.add_argument('--id', type=str, default='')

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1)

    parser.add_argument('--self_critical', type=int, default=0)
    # Used for self critical or structure. Used when sampling is need during training
    parser.add_argument('--train_sample_n', type=int, default=16)
    parser.add_argument('--train_sample_method', type=str, default='sample')
    parser.add_argument('--train_beam_size', type=int, default=1)

    # config
    parser.add_argument('--cfg', type=str, default=None)
    args = parser.parse_args()
    if args.cfg is not None:
        from misc.config import CfgNode
        cn = CfgNode.load_yaml_with_base(args.cfg)
        for k, v in cn.items():
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    args.checkpoint_path = 'log_{}'.format(args.id)

    # Deal with feature things before anything
    args.use_fc, args.use_att = True, False

    return args
