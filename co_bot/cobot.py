"""Training of COBOT chatbot : TO BE REFACTORED !"""
from torch import save

from co_bot.data_access.datasets import FriendsDataset
from co_bot.preprocessing.chains import TextSeqChain, PairChain
from co_bot.preprocessing.seq_modifiers import KeepOnlyCharacters, SeparateCharacters, Lowerize
from co_bot.preprocessing.pair_filters import IsNotTooLong
from co_bot.preprocessing.batch_builder import BatchBuilder
from co_bot.preprocessing.tokenizer import Tokenizer
from co_bot.model.encoder import Encoder

DATASETS = ((FriendsDataset, "../data/raw/friends_sequences.xlsx"), )

def main():

    # Build the full dataset
    full_dataset = []
    for dataset in DATASETS:
        dataset = dataset[0]().load_data(dataset[1])
        full_dataset.append(dataset)
    full_dataset = [pair for dataset in full_dataset for pair in dataset]

    # Apply text processing on any text sequence
    text_seq_chain = TextSeqChain(modifiers=(Lowerize(),
                                             SeparateCharacters(characters_to_separate=".!?'"),
                                             KeepOnlyCharacters(characters_to_keep="a-zA-Z.?!'")))

    full_dataset_preprocessed = []
    for pair in full_dataset:
        pair_preprocessed = []
        for sequence in pair:
            pair_preprocessed.append(text_seq_chain.apply(sequence))
        pair_preprocessed = tuple(pair_preprocessed)
        full_dataset_preprocessed.append(pair_preprocessed)
    del full_dataset

    # Apply filters on any pair of text sequences
    pair_chain = PairChain(filters=(IsNotTooLong(max_length=120),))
    full_dataset_preprocessed = pair_chain.apply_on_pairs(full_dataset_preprocessed)

    # Tokenize sequences
    tokenizer = Tokenizer()
    full_dataset_tokenized = []
    for pair in full_dataset_preprocessed:
        pair_tokenized = []
        for sequence in pair:
            pair_tokenized.append(tokenizer.process_sentence(sequence))
        pair_tokenized = tuple(pair_tokenized)
        full_dataset_tokenized.append(pair_tokenized)
    del full_dataset_preprocessed

    # Define the encoder
    vocabulary_size = tokenizer._size_vocabulary
    encoder = Encoder(nb_layers=1, hidden_size=50, vocabulary_size=vocabulary_size)

    # Process batch of training data
    full_dataset_tokenized_test = full_dataset_tokenized[0:10]
    batch_builder = BatchBuilder(training_data=full_dataset_tokenized_test)
    while True:
        input_sequences, input_lengths, output_sequences, output_lengths = batch_builder.get_batch(batch_size=10)
        if len(input_sequences) == 0:
            break
        output, hidden = encoder.forward(input_sequences, input_lengths)

    save(output, "../data/output.pt")
    save(hidden, "../data/hidden.pt")
    # To be continued...

if __name__ == '__main__':
    main()
