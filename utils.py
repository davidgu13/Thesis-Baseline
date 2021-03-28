import time
import math
import torch
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
MAX_INP_LENGTH = 34
MAX_OUT_LENGTH = 20


# misc = ['', ' ', '\xa0'] + list(string.punctuation + string.ascii_lowercase + string.ascii_uppercase + string.digits)
kat2eng = dict(zip(['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ'],
                   ['a', 'b', 'g', 'd', 'e', 'v', 'z', 't', 'i', "k'", 'l', 'm', 'n', 'o', "p'", 'ž', 'r', 's', "t'", 'u', 'p', 'k', 'ġ', "q'", 'š', 'č', 'c', 'j', "c'", "č'", 'x', 'ǰ', 'h']))
kat_alphabet = list(kat2eng.keys())
KAT_ALPHABET_SIZE = len(kat_alphabet)

morphfeat = ['PST','PRS','FUT','IMP','IND','SBJV','PFV','IPFV','PRF','COND','OPT','V.MSDR'] # based on the Screeves: 'IND;PRS','IND;IPFV','SBJV;PRS','IND;FUT','COND','SBJV;FUT','IND;PST;PFV','OPT','IND;PRF','IND;PST;PRF','SBJV;PRF','V.MSDR;PRF','V.MSDR;IPFV'
arguments_alph = ['SG','PL','1','2','3']
ENCODER_ALPHABET = ["<SOS>", "<EOS>", "<PAD>"] + kat_alphabet + arguments_alph + morphfeat
idx2enc_alph = dict(enumerate(ENCODER_ALPHABET))
ENC_ALPHABET_SIZE = len(ENCODER_ALPHABET)

enc_alph2idx = {y:x for x,y in idx2enc_alph.items()}

SOS_token, EOS_token, PAD_token = 0, 1, 2 # idx2enc_alph[SOS] = 'SOS', idx2enc_alph[EOS] = 'EOS', idx2enc_alph[PAD] = 'PAD'

IS_BIDIRECTIONAL = False


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def editDistDP(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] # Create a table to store results of subproblems
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, only option is to insert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j
            # If second string is empty, only option is to remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char and recur for remaining string
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace
    return dp[m][n]

def ED(s1,s2):
    return editDistDP(s1,s2,len(s1),len(s2))

# Unused commands from Preprocessing:
# vocab = build_lemmas_from_file(data_file)  # Not Final.txt") #{classes[4]}.txt")
# print(len(vocab))
# print(list(vocab.keys()))
# print(len(vocab)==len(set(vocab.keys())))
# print(sum([len(v) for v in vocab.values()]))
# samples, labels = [], []
# for paradigm in vocab.values():
#     ret1, ret2 = product(paradigm)
#     samples.extend(ret1)
#     labels.extend(ret2)
# assert len(samples) == len(labels)
# self.data_size = len(labels)
# print("Splitting to train & test sets...")
# X_train, X_test, y_train, y_test = train_test_split(samples[:int(dataset_usage * data_size)], labels[:int(dataset_usage * data_size)], test_size=0.2, random_state=42)  # Divide into train and test sets.
# samp_enc, label_enc = sample2vector(X_train[0]), label2vector(y_train[0])
# assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
# print(f"Train set size: {len(X_train)}, test set size = {len(X_test)}")
# print("Encoding data as 1-hot vectors...")
# X_train, X_test, y_train, y_test = encode_samples(X_train), encode_samples(X_test), encode_labels(y_train), encode_labels(y_test)
# self.X = samples
# self.y = labels
# self.pairs_train = list(zip(X_train, y_train))
# self.pairs_test = list(zip(X_test, y_test))
