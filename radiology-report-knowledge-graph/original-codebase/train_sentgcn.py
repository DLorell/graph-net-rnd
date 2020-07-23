from data import Biview_MultiSent, sent_collate_fn
from sentgcn import SentGCN
from evaluate import evaluate
from my_build_vocab import Vocabulary
import os
import math
import argparse
import logging
import pickle
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default='/share/project/modimnet/gcnmodels')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset-dir', type=str, default='data')
    parser.add_argument('--train-folds', type=str, default='012')
    parser.add_argument('--val-folds', type=str, default='3')
    parser.add_argument('--test-folds', type=str, default='4')
    parser.add_argument('--report-path', type=str, default='data/reports.json')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl')
    parser.add_argument('--label-path', type=str, default='data/label_dict.json')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--log-freq', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--encoder-lr', type=float, default=1e-6)
    parser.add_argument('--decoder-lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--clip-value', type=float, default=5.0)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    args.model_dir += "/"+args.name
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.log_dir, args.name + '.log'), level=logging.INFO)
    print('------------------------Model and Training Details--------------------------')
    print(args)
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))

    writer = SummaryWriter(log_dir=os.path.join('runs', args.name))

    gpus = [int(_) for _ in list(args.gpus)]
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    train_set = Biview_MultiSent('train', args.dataset_dir, args.train_folds, args.report_path, args.vocab_path, args.label_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=sent_collate_fn)
    val_set = Biview_MultiSent('val', args.dataset_dir, args.val_folds, args.report_path, args.vocab_path, args.label_path)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=sent_collate_fn)
    test_set = Biview_MultiSent('test', args.dataset_dir, args.test_folds, args.report_path, args.vocab_path, args.label_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=sent_collate_fn)

    fw_adj = torch.tensor([
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float, device=device)
    bw_adj = fw_adj.t()

    model = SentGCN(args.num_classes, fw_adj, bw_adj, len(vocab)).to(device)
    for param in model.densenet121.parameters():
        param.requires_grad = False
    for param in model.cls_atten.parameters():
        param.requires_grad = False
    for param in model.gcn.parameters():
        param.requires_grad = False
    model.densenet121.eval()
    model.gcn.eval()
    decoder_params = \
        list(model.atten.parameters()) + \
        list(model.embed.parameters()) + \
        list(model.init_sent_h.parameters()) + \
        list(model.init_sent_c.parameters()) + \
        list(model.sent_lstm.parameters()) + \
        list(model.word_lstm.parameters()) + \
        list(model.fc.parameters())

    CELoss = nn.CrossEntropyLoss(reduction='none')

    gamma=0.1
    optimizer = torch.optim.Adam(decoder_params, lr=args.decoder_lr*(1/gamma), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=gamma)
    scheduler.step(0)

    if args.pretrained:
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['model_state_dict']
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in pretrained_state_dict.items() if k in state_dict and 'fc' not in k})
        model.load_state_dict(state_dict)

    start_epoch = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    num_steps = math.ceil(len(train_set) / args.batch_size)

    val_gts = {}
    test_gts = {}

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.dropout.train()
        epoch_loss = 0.0
        print('------------------------Training for Epoch {}---------------------------'.format(epoch))
        print('Learning rate {:.7f}'.format(optimizer.param_groups[0]['lr']))

        for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(train_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            captions = captions.to(device)
            loss_masks = loss_masks.to(device).to(torch.bool)
            update_masks = update_masks.to(device).to(torch.bool)
            optimizer.zero_grad()
            logits = model(images1, images2, captions[:, :, :-1], update_masks)
            logits = logits.permute(0, 3, 1, 2).contiguous()
            captions = captions[:, :, 1:].contiguous()
            loss_masks = loss_masks[:, :, 1:].contiguous()
            loss = CELoss(logits, captions)
            loss = loss.masked_select(loss_masks).mean()
            loss.backward()
            epoch_loss += loss.item()
            clip_grad_value_(model.parameters(), args.clip_value)
            optimizer.step()

            print("{:.4f}% done...".format(100 * ((i+1)/len(train_loader))), end="\r")

        epoch_loss /= num_steps
        print('Epoch {}/{}, Loss {:.4f}'.format(epoch, args.num_epochs, epoch_loss))
        writer.add_scalar('loss', epoch_loss, epoch)

        scheduler.step(epoch)

        if epoch % args.log_freq == 0:

            save_fname = os.path.join(args.model_dir, '{}_e{}.pth'.format(args.name, epoch))
            if len(gpus) > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_fname)

            model.dropout.eval()
            val_res = {}
            for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(val_loader):
                images1 = images1.to(device)
                images2 = images2.to(device)
                preds = model(images1, images2, stop_id=vocab('.'))

                caseid = caseids[0]
                val_res[caseid] = ['']
                pred = preds[0].detach().cpu()
                for isent in range(pred.size(0)):
                    words = []
                    for wid in pred[isent].tolist():
                        w = vocab.idx2word[wid]
                        if w == '<start>' or w == '<pad>':
                            continue
                        if w == '<end>':
                            break
                        words.append(w)
                    val_res[caseid][0] += ' '.join(words)
                    val_res[caseid][0] += ' '

                if epoch == start_epoch:
                    val_gts[caseid] = ['']
                    cap = captions[0]
                    for isent in range(cap.size(0)):
                        words = []
                        for wid in cap[isent, 1:].tolist():
                            w = vocab.idx2word[wid]
                            if w == '<start>' or w == '<pad>':
                                continue
                            if w == '<end>':
                                break
                            words.append(w)
                        val_gts[caseid][0] += ' '.join(words)
                        val_gts[caseid][0] += ' '

            if val_gts.keys() != val_res.keys():
                print("val_gts: ")
                print(val_gts.keys())
                print("val_res: ")
                print(val_res.keys())
                print('Diff: ')
                gts_set = set(list(val_gts.keys()))
                res_set = set(list(val_res.keys()))
                print(res_set.difference(gts_set))
                assert 0==1

            scores = evaluate(val_gts, val_res)
            writer.add_scalar('VAL BLEU 1', scores['Bleu_1'], epoch)
            writer.add_scalar('VAL BLEU 2', scores['Bleu_2'], epoch)
            writer.add_scalar('VAL BLEU 3', scores['Bleu_3'], epoch)
            writer.add_scalar('VAL BLEU 4', scores['Bleu_4'], epoch)
            writer.add_scalar('VAL ROUGE_L', scores['ROUGE_L'], epoch)
            writer.add_scalar('VAL CIDEr', scores['CIDEr'], epoch)
            # writer.add_scalar('VAL Meteor', scores['METEOR'], epoch)

            test_res = {}
            for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(test_loader):
                images1 = images1.to(device)
                images2 = images2.to(device)
                preds = model(images1, images2, stop_id=vocab('.'))

                caseid = caseids[0]
                test_res[caseid] = ['']
                pred = preds[0].detach().cpu()
                for isent in range(pred.size(0)):
                    words = []
                    for wid in pred[isent].tolist():
                        w = vocab.idx2word[wid]
                        if w == '<start>' or w == '<pad>':
                            continue
                        if w == '<end>':
                            break
                        words.append(w)
                    test_res[caseid][0] += ' '.join(words)
                    test_res[caseid][0] += ' '

                if epoch == start_epoch:
                    test_gts[caseid] = ['']
                    cap = captions[0]
                    for isent in range(cap.size(0)):
                        words = []
                        for wid in cap[isent, 1:].tolist():
                            w = vocab.idx2word[wid]
                            if w == '<start>' or w == '<pad>':
                                continue
                            if w == '<end>':
                                break
                            words.append(w)
                        test_gts[caseid][0] += ' '.join(words)
                        test_gts[caseid][0] += ' '

            scores = evaluate(test_gts, test_res)
            writer.add_scalar('TEST BLEU 1', scores['Bleu_1'], epoch)
            writer.add_scalar('TEST BLEU 2', scores['Bleu_2'], epoch)
            writer.add_scalar('TEST BLEU 3', scores['Bleu_3'], epoch)
            writer.add_scalar('TEST BLEU 4', scores['Bleu_4'], epoch)
            writer.add_scalar('TEST ROUGE_L', scores['ROUGE_L'], epoch)
            writer.add_scalar('TEST CIDEr', scores['CIDEr'], epoch)
            # writer.add_scalar('TEST Meteor', scores['METEOR'], epoch)

            with open(os.path.join(args.output_dir, '{}_test_e{}.csv'.format(args.name, epoch)), 'w') as f1:
                with open(os.path.join(args.output_dir, '{}_test_gts.csv'.format(args.name)), 'w') as f2:
                    for caseid in test_res.keys():
                        f1.write(test_res[caseid][0] + '\n')
                        f2.write(test_gts[caseid][0] + '\n')

    writer.close()


    validate(args, start_epoch, gpus, model, optimizer, device, val_loader, test_loader, vocab, start_epoch, val_gts, test_gts)



def validate(args, epoch, gpus, model, optimizer, device, val_loader, test_loader, vocab, start_epoch, val_gts, test_gts):

    print("\n\n\n")

    iterations = 3

    model.dropout.eval()
    val_res = {}
    for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(val_loader):

        if i > iterations:
            break

        images1 = images1.to(device)
        images2 = images2.to(device)
        preds = model(images1, images2, stop_id=vocab('.'))

        caseid = caseids[0]
        val_res[caseid] = ['']
        pred = preds[0].detach().cpu()
        for isent in range(pred.size(0)):
            words = []
            for wid in pred[isent].tolist():
                w = vocab.idx2word[wid]
                if w == '<start>' or w == '<pad>':
                    continue
                if w == '<end>':
                    break
                words.append(w)
            val_res[caseid][0] += ' '.join(words)
            val_res[caseid][0] += ' '

        if epoch == start_epoch:
            val_gts[caseid] = ['']
            cap = captions[0]
            for isent in range(cap.size(0)):
                words = []
                for wid in cap[isent, 1:].tolist():
                    w = vocab.idx2word[wid]
                    if w == '<start>' or w == '<pad>':
                        continue
                    if w == '<end>':
                        break
                    words.append(w)
                val_gts[caseid][0] += ' '.join(words)
                val_gts[caseid][0] += ' '

    if val_gts.keys() != val_res.keys():
        print("val_gts: ")
        print(val_gts.keys())
        print("val_res: ")
        print(val_res.keys())
        print('Diff: ')
        gts_set = set(list(val_gts.keys()))
        res_set = set(list(val_res.keys()))
        print(res_set.difference(gts_set))
        assert 0==1

    scores = evaluate(val_gts, val_res)

    """
    writer.add_scalar('VAL BLEU 1', scores['Bleu_1'], epoch)
    writer.add_scalar('VAL BLEU 2', scores['Bleu_2'], epoch)
    writer.add_scalar('VAL BLEU 3', scores['Bleu_3'], epoch)
    writer.add_scalar('VAL BLEU 4', scores['Bleu_4'], epoch)
    writer.add_scalar('VAL ROUGE_L', scores['ROUGE_L'], epoch)
    writer.add_scalar('VAL CIDEr', scores['CIDEr'], epoch)
    # writer.add_scalar('VAL Meteor', scores['METEOR'], epoch)
    """

    test_res = {}
    for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(test_loader):

        if i > iterations:
            break

        images1 = images1.to(device)
        images2 = images2.to(device)
        preds = model(images1, images2, stop_id=vocab('.'))

        caseid = caseids[0]
        test_res[caseid] = ['']
        pred = preds[0].detach().cpu()
        for isent in range(pred.size(0)):
            words = []
            for wid in pred[isent].tolist():
                w = vocab.idx2word[wid]
                if w == '<start>' or w == '<pad>':
                    continue
                if w == '<end>':
                    break
                words.append(w)
            test_res[caseid][0] += ' '.join(words)
            test_res[caseid][0] += ' '

        if epoch == start_epoch:
            test_gts[caseid] = ['']
            cap = captions[0]
            for isent in range(cap.size(0)):
                words = []
                for wid in cap[isent, 1:].tolist():
                    w = vocab.idx2word[wid]
                    if w == '<start>' or w == '<pad>':
                        continue
                    if w == '<end>':
                        break
                    words.append(w)
                test_gts[caseid][0] += ' '.join(words)
                test_gts[caseid][0] += ' '

    scores = evaluate(test_gts, test_res)
    """
    writer.add_scalar('TEST BLEU 1', scores['Bleu_1'], epoch)
    writer.add_scalar('TEST BLEU 2', scores['Bleu_2'], epoch)
    writer.add_scalar('TEST BLEU 3', scores['Bleu_3'], epoch)
    writer.add_scalar('TEST BLEU 4', scores['Bleu_4'], epoch)
    writer.add_scalar('TEST ROUGE_L', scores['ROUGE_L'], epoch)
    writer.add_scalar('TEST CIDEr', scores['CIDEr'], epoch)
    # writer.add_scalar('TEST Meteor', scores['METEOR'], epoch)
    """

    """
    with open(os.path.join(args.output_dir, '{}_test_e{}.csv'.format(args.name, epoch)), 'w') as f1:
        with open(os.path.join(args.output_dir, '{}_test_gts.csv'.format(args.name)), 'w') as f2:
            for caseid in test_res.keys():
                f1.write(test_res[caseid][0] + '\n')
                f2.write(test_gts[caseid][0] + '\n')
    """


if __name__ == '__main__':
    main()