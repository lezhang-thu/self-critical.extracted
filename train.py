import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import eval_utils
import misc.utils as utils
import models
import opts
from dataloader import *
from misc.loss_wrapper import LossWrapper
from misc.rewards import init_scorer


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'vocab': loader.get_vocab(),
    }
    # Load old infos (if there is) and check if models are compatible
    if opt.checkpoint_path is not None and os.path.isfile(
            os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            print('infos load success')
    infos['opt'] = opt

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab
    # Load pretrained weights:
    if opt.checkpoint_path is not None and os.path.isfile(os.path.join(opt.checkpoint_path, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'model.pth')))
        print('model load success')

    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    # Wrap with dataparallel
    dp_model = torch.nn.DataParallel(model)
    dp_lw_model = torch.nn.DataParallel(lw_model)

    ##########################
    #  Build optimizer
    ##########################
    optimizer = utils.ReduceLROnPlateau(optim.Adam(model.parameters(), opt.learning_rate),
            factor=0.5, patience=3)
    # Load the optimizer
    if opt.checkpoint_path is not None and os.path.isfile(os.path.join(opt.checkpoint_path, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    best_val_score = infos.get('best_val_score', None)
    print('iter {}, epoch {}, best_val_score {}'.format(
        iteration, epoch, best_val_score))

    print(sorted(dict(set(vars(opt).items())).items(), key=lambda x: x[0]))
    # Start training
    if opt.self_critical:
        init_scorer(opt.cached_tokens)
    # Assure in training mode
    dp_lw_model.train()
    try:
        while True:
            # Stop if reaching max_epoch
            if epoch >= opt.max_epochs:
                break

            # Load data from train split (0)
            data = loader.get_batch('train')

            torch.cuda.synchronize()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                    torch.arange(0, len(data['gts'])))

            loss = model_out['loss'].mean()

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1

            # Write the training loss summary
            if iteration % opt.losses_log_every == 0:
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                if opt.self_critical:
                    tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch

            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                tb_summary_writer.add_scalar('epoch', epoch, iteration)
                # eval model
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                _, _, lang_stats = eval_utils.eval_split(
                    dp_model, loader, eval_kwargs)

                optimizer.scheduler_step(-lang_stats['CIDEr'])
                # Write validation result into summary
                for k, v in lang_stats.items():
                    tb_summary_writer.add_scalar(k, v, iteration)

                # Save model if is improving on validation result
                current_score = lang_stats['CIDEr']

                best_flag = False
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscellaneous information
                infos['best_val_score'] = best_val_score

                utils.save_checkpoint(opt, model, infos, optimizer)
                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

    except (RuntimeError, KeyboardInterrupt):
        pass


opt = opts.parse_opt()
train(opt)
