from Network import *
from training import *
from torch.utils.data import random_split
from torch import save
import torch
from numpy import inf
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--limit_dataset_size", help="Whether to limit the dataset size.", action="store_true")
    parser.add_argument('-d', "--dataset_size", help="The number of samples to take from the datset. Must not be larger than 540,000!", type=int)
    # parser.add_argument('-e', "--embedding_size", help="The Embedding vectors dimension.", type=int) # not used, for now emb_size=hid_size
    parser.add_argument("hidden_size", help="The hidden vectors size in the LSTMs.", type=int)
    parser.add_argument("epochs", help="Number of epochs to iterate", type=int)
    parser.add_argument("batch_size", help="The batch size", type=int)
    args = parser.parse_args()

    # classes = ['Transitive', 'Intransitive', 'Medial', 'Indirect', 'Stative']
    # Set hyper-parameters
    t_start = time.time()
    reinflection_data_file_name = f"Reinflection Data_kat_indexes.txt"
    limit_dataset_size = args.limit_dataset_size
    dataset_size = args.dataset_size
    # embedding_size = args.embedding_size # 80
    hidden_size = args.hidden_size # 256
    epochs = args.epochs # 2
    batch_size = args.batch_size # 200
    test_set_ratio = 0.3 # make bigger than 0.001, and do prints only for random 20 ones.
    lr = 0.01
    torch.manual_seed(42)


    print("Reading data & constructing Reinflection format... ",end='')
    t0 = time.time()
    dataset = ReinflectionDataset(reinflection_data_file_name, dataset_size, limit_dataset_size)
    print(f"took {time.time()-t0} seconds")

    test_set_size = int(test_set_ratio*len(dataset))
    train_set_size = len(dataset) - test_set_size
    train_set, test_set = random_split(dataset, [train_set_size, test_set_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=my_collate)

    print("Creating the model...")
    encoder1 = EncoderLSTM(ENC_ALPHABET_SIZE, hidden_size, hidden_size).to(device)
    attn_decoder1 = AttnDecoderLSTM(hidden_size, KAT_ALPHABET_SIZE+2, dropout_p=0.1).to(device)

    print("Starting the training process:")
    plot_losses = []
    plot_distances = []
    ed = 0.0
    for e in range(epochs):
        print(f"Started epoch {e+1}:")
        for i_batch, batch in enumerate(train_dataloader):
            print(f"Batch {i_batch}")
            encoder1.train()
            attn_decoder1.train()
            plot_losses = trainIters(batch, encoder1, attn_decoder1, batch_size, plot_losses, print_every=100, learning_rate=lr)

            # Evaluation

            if (i_batch+1)%10==0:
                encoder1.eval()
                attn_decoder1.eval()
                ed = 0.0
                print('')
                for (test_sample, test_label) in test_dataloader:
                    ed += evaluateRandomly((test_sample, test_label), encoder1, attn_decoder1, (i_batch+1)%1000==0)
                avg = ed / test_set_size
                print(avg)
                plot_distances.append(ed / test_set_size)
                if avg==0.0: break

    with open("saved encoder.pt", 'wb') as f:
        save(encoder1,f)

    with open("saved decoder.pt", 'wb') as f:
        save(attn_decoder1,f)

    print(f"total runtime = {time.time() - t_start}")

    plt.figure()
    plt.subplot(211)
    plt.plot(plot_losses)
    plt.subplot(212)
    plt.plot(plot_distances)
    plt.show()


    # For next time: see if these commands do not raise the warnings (using state_dict instead):
    # with open("saved encoder.pt", 'wb') as f:
    #     save({'state_dict': encoder1.state_dict()},f)
    #
    # with open("saved decoder.pt", 'wb') as f:
    #     save({'state_dict': attn_decoder1.state_dict()},f)

    # encdec_net = BasicNetwork(input_size=ENC_ALPHABET_SIZE, embed_size=30, hid_dim=50,
    # output_size=KAT_ALPHABET_SIZE)
    # Encode and build a basic network - Embedding + LSTM.

    # Train and measure error with Edit distance.
