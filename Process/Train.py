import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Utils.transform import get_transform
from Utils.data_utils import ImageList
from Utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from loss.mmd_loss import MMD_loss
from .Metrics import AverageMeter
from .Valid import valid


logger = None

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def train(args, model):
    global logger
    logger = args.logger

    """ Train the model """
    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size
    transform_train, transform_test = get_transform(args.img_size)

    train_data_list = open(args.train_list).readlines()
    test_data_list = open(args.test_list).readlines()

    train_loader = torch.utils.data.DataLoader(
        ImageList(train_data_list, args.dataset_path, transform=transform_train, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    target_loader = torch.utils.data.DataLoader(
        ImageList(test_data_list, args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        ImageList(test_data_list, args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    main_loader, secondary_loader, main_is_source = ((train_loader, target_loader, True)
                                                     if len(train_data_list) > len(test_data_list)
                                                     else (target_loader, train_loader, False))

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * 1)

    model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    loss_mmd = MMD_loss()
    loss_fct = torch.nn.CrossEntropyLoss()
    while True:
        model.train()
        epoch_iterator = tqdm(main_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        secondary_iterator = iter(secondary_loader)
        for step, batch in enumerate(epoch_iterator):
            try:
                secondary_target = next(secondary_iterator)
            except StopIteration:
                secondary_iterator = iter(secondary_loader)
                secondary_target = next(secondary_iterator)

            if main_is_source:
                batch_source = tuple(t.to(args.device) for t in batch)
                x_source, y_source = batch_source

                batch_target = tuple(t.to(args.device) for t in secondary_target)
                x_target, y_target = batch_target
            else:
                batch_source = tuple(t.to(args.device) for t in secondary_target)
                x_source, y_source = batch_source

                batch_target = tuple(t.to(args.device) for t in batch)
                x_target, y_target = batch_target

            source_features, logits = model(x_source)

            # CE Loss
            loss = loss_fct(logits.view(-1, model.head.out_features), y_source.view(-1))

            if not args.source_only:
                target_features, _ = model(x_target)
                loss_mmd_cal = loss_mmd(source_features, target_features)

                loss = (0.7 * loss) + (0.3 * loss_mmd_cal)

            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
            if global_step % args.eval_every == 0:
                accuracy, cacc = valid(args, model, test_loader, global_step)

                logger.info(f"Accuracy: {accuracy}")

                if best_acc < accuracy:
                    save_model(args, model)
                    best_acc = accuracy
                model.train()

            if global_step % t_total == 0:
                break

        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")