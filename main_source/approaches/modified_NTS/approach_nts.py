import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import DataParallel
from datetime import datetime
import main_source.utils.load_data as load_data
import main_source.utils.utils as utils
import main_source.utils.check_point as ckp


# init best prec for saving checkpoint
best_val_acc = 0


def approach_process(config):
    global best_val_acc, _print
    dict_config = config

    # config save log and model dir
    save_dir = os.path.join(dict_config['save_model_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = utils.init_log(save_dir)
    _print = logging.info

    # Load data
    train_loader, val_loader = load_data.load_data_from_csv(dict_config)

    # define model
    if dict_config["model"] == 'NTS_resnet':
        import main_source.models.NTS.model as md
        net = md.attention_net(topN=dict_config['proposal_num'], num_classes=dict_config['num_classes'], cat_num=dict_config['cat_num'],
                               input_shape = (dict_config['size_height'], dict_config['size_width']))

    # Load model from resume
    if dict_config["resume_pretrained"]:
        ckpt = torch.load(dict_config['link_pretrained'])
        net.load_state_dict(ckpt['net_state_dict'])
        dict_config["start_epoch"] = ckpt['epoch'] + 1

    # set creterion
    creterion = torch.nn.CrossEntropyLoss()

    # define optimizers
    raw_parameters = list(net.pretrained_model.parameters())
    part_parameters = list(net.proposal_net.parameters())
    concat_parameters = list(net.concat_net.parameters())
    partcls_parameters = list(net.partcls_net.parameters())

    raw_optimizer = torch.optim.SGD(raw_parameters, lr=dict_config["lr"], momentum=0.9, weight_decay=dict_config["wd"])
    concat_optimizer = torch.optim.SGD(concat_parameters, lr=dict_config["lr"], momentum=0.9, weight_decay=dict_config["wd"])
    part_optimizer = torch.optim.SGD(part_parameters, lr=dict_config["lr"], momentum=0.9, weight_decay=dict_config["wd"])
    partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=dict_config["lr"], momentum=0.9, weight_decay=dict_config["wd"])
    schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
    net = net.cuda()
    net = DataParallel(net)

    for epoch in range(dict_config["start_epoch"], dict_config["epochs"]):
        for scheduler in schedulers:
            scheduler.step()
        train_batch(net, train_loader, (raw_optimizer, part_optimizer, concat_optimizer, partcls_optimizer), creterion,
                    dict_config['proposal_num'])

        if epoch % dict_config['val_freq'] == 0:
            val_acc = val(net, val_loader, creterion, epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckp.save_model(net.module.state_dict(), epoch, best_val_acc, dict_config['save_model_dir'],
                               'best_model.ckpt')

        if epoch % dict_config['save_checkpoint_freq'] == 0:
            ckp.save_model(net.module.state_dict(), epoch, best_val_acc, dict_config['save_model_dir'],
                           'checkpoint_%03d.ckpt' % epoch)


def train_batch(net, train_loader, optimizer, creterion, proposal_num):
    net.train()
    for i, (X, y) in enumerate(train_loader):
        img = X.to(torch.device("cuda"))
        label = y.to(torch.device("cuda"))

        batch_size = img.size(0)
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()
        optimizer[3].zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)

        part_loss = utils.list_loss(part_logits.view(batch_size * proposal_num, -1),
                                    label.unsqueeze(1).repeat(1, proposal_num).view(-1)).view(batch_size, proposal_num)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = utils.ranking_loss(top_n_prob, part_loss, proposal_num)
        partcls_loss = creterion(part_logits.view(batch_size * proposal_num, -1),
                                 label.unsqueeze(1).repeat(1, proposal_num).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        optimizer[0].step()
        optimizer[1].step()
        optimizer[2].step()
        optimizer[3].step()

        utils.progress_bar(i, len(train_loader), 'train')


def val(net, validate_loader, creterion, epoch):
    val_loss = 0
    val_correct = 0
    total = 0
    net.eval()
    for i, data in enumerate(validate_loader):
        with torch.no_grad():
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            _, concat_logits, _, _, _ = net(img)
            # calculate loss
            concat_loss = creterion(concat_logits, label)
            # calculate accuracy
            _, concat_predict = torch.max(concat_logits, 1)
            total += batch_size
            val_correct += torch.sum(concat_predict.data == label.data)
            val_loss += concat_loss.item() * batch_size
            utils.progress_bar(i, len(validate_loader), 'eval test set')

    val_acc = float(val_correct) / total
    val_loss = val_loss / total
    _print(
        'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
            epoch,
            val_loss,
            val_acc,
            total))

    return val_acc
