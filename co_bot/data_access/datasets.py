"""Implementation of datasets"""
from ast import literal_eval
import re

import pandas as pd

from co_bot.data_access.interfaces import IDataset


class FriendsDataset(IDataset):
    """Friends Dataset that contains dialogues from the TV show

    """
    def __init__(self):
        super(IDataset, self).__init__()

    def load_data(self, path: str):
        """Load Friends dataset

        Args:
            path (str): path to friends dataset at xslx format

        Returns:
            sequences (Sequence[Tuple[str, str]]): sequence of dialogues (input - output) between friends characters

        """
        df_friends = pd.read_excel(path)
        input_sequence = df_friends['~~Input Dialog~~'].tolist()
        output_sequence = df_friends['~~Output Dialog~~'].tolist()
        dataset = [sequence for sequence in zip(input_sequence, output_sequence)]

        return dataset


class CornellMovieDataset(IDataset):
    """Cornell Movie Dataset that contains 220,579 conversational exchanges involving 9,035 characters from 617 movies

    """
    def load_data(self, path_movie_conversations: str, path_movie_lines: str):
        """Load Cornell movie corpus dataset

        Args:
            path_movie_conversations (str): path to txt file that contains the structure of the conversations
            path_movie_lines (str): path to txt file that contains the actual text of each utterance

        Returns:
            sequences (Sequence[Tuple[str, str]]): sequence of dialogues (input - output) between movies characters

        """
        movie_conversations = pd.read_csv(path_movie_conversations,
                                          sep=r'\+\+\+\$\+\+\+',
                                          names=['character_ID_1',
                                                 'character_ID_2',
                                                 'movie_ID',
                                                 'sequence_utterances'])['sequence_utterances'].values.tolist()

        movie_lines = pd.read_csv(path_movie_lines,
                                     sep=r'\+\+\+\$\+\+\+',
                                     names=['line_ID',
                                            'character_ID',
                                            'movie_ID',
                                            'character_name',
                                            'text_utterance'])[['line_ID', 'text_utterance']]

        movie_lines['line_ID'] = movie_lines['line_ID'].apply(lambda x: re.sub("\s+", "", x))
        movie_lines.set_index('line_ID', inplace=True)

        conversations = [literal_eval(re.sub("\s+", "", sequence)) for sequence in movie_conversations]
        conversations = [list(zip(sequence[::1], sequence[1::1])) for sequence in conversations]
        conversations = [dialogue for sequence in conversations for dialogue in sequence]
        dataset = [(movie_lines.loc[dialogue[0]]['text_utterance'],
                              movie_lines.loc[dialogue[1]]['text_utterance']) for dialogue in conversations]

        return dataset


class SimpsonsDataset(IDataset):
    """Simpsons Dataset that contains the dialogues of the TV show from 27 seasons

    """
    def load_data(self, path: str):
        """Load Simpsons dataset

        Args:
            path (str): path to simpsons dataset at csv format

        Returns:
            sequences (Sequence[Tuple[str, str]]): sequence of dialogues (input - output) between Simpsons characters

        """
        df_simpsons = pd.read_csv(path)
        df_simpsons.fillna("#####", inplace=True)
        conversations = df_simpsons['spoken_words'].to_list()
        dialogues = []
        dialogue = []
        for sentence in conversations:
            if sentence != "#####":
                dialogue.append(sentence)
            else:
                dialogues.append(dialogue)
                dialogue = []

        conversations = [list(zip(sequence[::1], sequence[1::1])) for sequence in dialogues]
        dataset = [(sequence[0], sequence[1]) for conversation in conversations for sequence in conversation]

        return dataset


if __name__ == "__main__":
    # For testing purposes, to be removed
    friends_dataset = FriendsDataset().load_data(path="../../data/raw/friends_sequences.xlsx")
    print(friends_dataset[100])
    cornell_movie = CornellMovieDataset().\
        load_data(path_movie_conversations="../../data/raw/cornell_movie_dialogs_corpus/movie_conversations.txt",
                  path_movie_lines="../../data/raw/cornell_movie_dialogs_corpus/movie_lines.txt")
    print(cornell_movie[100])
    simpsons_dataset = SimpsonsDataset().load_data(path="../../data/raw/simpsons_dataset.csv")
    print(simpsons_dataset[100])
