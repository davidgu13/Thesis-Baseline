from preprocessing import *
import random
import torch.optim as optim
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
import torch.nn as nn
from utils import ED
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

teacher_forcing_ratio = 0.5

def train(batch, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_INP_LENGTH):
    # input_tensor, input_lens = pad_packed_sequence(input_tensor)
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    # input_length = input_tensor.size(0)
    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0, 0]
    # Use here the
    pass
    # target_tensor = label_batch_tensor

    x_padded, y_padded, x_lens, y_lens = batch

    encoder_output_padded, encoder_output_lens, encoder_hidden = encoder(x_padded, x_lens, encoder_hidden)
    encoder_outputs = encoder_output_padded[0]

    decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device).t() #.unsqueeze(-1) # hope it won't break anything
    decoder_hidden = encoder_hidden


    use_teacher_forcing = random.random() < teacher_forcing_ratio
    target_length = y_padded.size(0)
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            loss += criterion(decoder_output, y_padded[di])
            decoder_input = y_padded[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, y_padded[di])
            if decoder_input.item() == EOS_token: # hope it won't break anything
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def trainIters(batch, encoder, decoder, batch_size, plot_losses, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs)) for _ in range(n_iters)]
    criterion = nn.NLLLoss()
    # print(f"This single epoch will consist of {n_iters} steps.")

    # for k in range(1, n_iters + 1):
        # training_pair = training_data[k - 1]
    # input_tensor = batch_samples[k-1]
    # target_tensor = batch_labels[k-1]

    loss = train(batch, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    if k % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, k / n_iters), k, k / n_iters * 100, print_loss_avg))

    if k % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

    showPlot(plot_losses)
    return plot_losses


def showPlot(points):
    plt.figure()
    # fig, ax = plt.subplots()
    # loc = ticker.MultipleLocator(base=0.2) # this locator puts ticks at regular intervals
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    # plt.pause(0.5)


def evaluate(encoder, decoder, X_test_single, max_length=MAX_INP_LENGTH):
    with torch.no_grad():
        # input_tensor = tensorFromSentence(input_lang, sentence)
        input_tensor = X_test_single
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(idx2enc_alph[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(X_and_y, encoder, decoder, verbose):
    # for i in range(n):
        # pair = random.choice(X_and_y)
    pair = X_and_y
    if verbose: print('>', vector2sample(pair[0][0]))
    if verbose: print('=', vector2sample(pair[1][0]))
    output_words, attentions = evaluate(encoder, decoder, pair[0][0].t())
    output_sentence = ' '.join(output_words)
    ed = ED("".join(output_words),"".join(vector2sample(pair[1][0])))
    if verbose: print(f'< {output_sentence} - ED(target,pred) = {ed}')
    if verbose: print('')
    return ed
    # For next time - add in the main a list for storing avg ED on test set!
    # On the Entire dataset, there should be a decreasing graph.
