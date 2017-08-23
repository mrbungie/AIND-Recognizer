import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    test_set= sorted([(item[0], item[1]) for item in test_set.get_all_Xlengths().items()], key=lambda x: x[0])
    probabilities = []
    guesses = []
    for word_id, test_Xlength in test_set:
        probability_dict = dict()
        for word, model in models.items():
            try:
                probability_dict[word] = model.score(test_Xlength[0], test_Xlength[1])
            except:
                continue
        probabilities.append(probability_dict)
        guesses.append(max(probability_dict, key=probability_dict.get))
    return probabilities, guesses
