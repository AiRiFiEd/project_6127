import logging
from ..static import LOGGER_PRINT_LEVEL
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_PRINT_LEVEL)

from pprint import pprint

import torch
from torch.utils.data import DataLoader

# Mask variable
def _mask(prev_generated_seq, config):
    prev_mask = torch.eq(prev_generated_seq, 1).type(torch.float32)
    lengths = torch.argmax(prev_mask,dim=1)
    max_len = prev_generated_seq.size(1)
    mask = []
    for i in range(prev_generated_seq.size(0)):
        if lengths[i] == 0:
            mask_line = [0] * max_len
        else:
            mask_line = [0] * lengths[i].item()
            mask_line.extend([1] * (max_len - lengths[i].item()))
        mask.append(mask_line)
    mask = torch.ByteTensor(mask)
    if config.cuda_is_available:
        mask = mask.cuda()
    return prev_generated_seq.data.masked_fill_(mask, 0.)

def train_batch(model, criterion, optimizer, vocab_size,
                input_variable, input_lengths, target_variable, 
                teacher_forcing_ratio, config):
    loss_list = []
    # Forward propagation
    prev_generated_seq = None
    target_variable_reshaped = target_variable[:, 1:].contiguous().view(-1)

    for i in range(config.num_of_exams):
        decoder_outputs, _, other = \
            model(input_variable, prev_generated_seq, input_lengths,
                   target_variable, teacher_forcing_ratio)

        decoder_outputs_reshaped = decoder_outputs.view(-1, vocab_size)
        lossi = criterion(decoder_outputs_reshaped, target_variable_reshaped)
        loss_list.append(lossi.item())
        model.zero_grad()
        lossi.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        prev_generated_seq = torch.squeeze(torch.topk(decoder_outputs, 1, dim=2)[1]).view(-1, decoder_outputs.size(1))
        prev_generated_seq = _mask(prev_generated_seq, config)
    return loss_list


def train_epoches(dataset, model, criterion, optimizer, 
                    vocab_size, teacher_forcing_ratio, 
                    config):
    # train_loader = DataLoader(dataset, config.batch_size)
    train_loader = dataset
    model.train(True)
    prev_epoch_loss_list = [100] * config.num_of_exams
    for epoch in range(1, config.epochs + 1):
        epoch_examples_total = 0
        epoch_loss_list = [0] * config.num_of_exams
        for batch_idx, (source, target, input_lengths) in enumerate(train_loader):
            try:
                input_variables = source
                target_variables = target
                # train model              
                loss_list = train_batch(model, criterion, optimizer, vocab_size,
                                            input_variables, input_lengths.tolist(),
                                            target_variables, teacher_forcing_ratio,
                                            config)
                # Record average loss
                num_examples = len(source)
                epoch_examples_total += num_examples
                for i in range(config.num_of_exams):
                    epoch_loss_list[i] += loss_list[i] * num_examples
            except:
                continue

        for i in range(config.num_of_exams):
            epoch_loss_list[i] /= float(epoch_examples_total)

        log_msg = "Finished epoch %d with losses:" % epoch
        logger.debug(log_msg)
        pprint(epoch_loss_list)
        if prev_epoch_loss_list[:-1] < epoch_loss_list[:-1]:
            break
        else:
            prev_epoch_loss_list = epoch_loss_list[:]
