from sentence_transformers import SentenceTransformer, util


class SimilarityChecker:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def similarity(self, s1, s2):
        query1, query2 = self.model.encode(s1), self.model.encode(s2)
        return util.cos_sim(query1, query2).item()

if __name__ == "__main__":
    sentences = ['Uchodźcy z Ukrainy będą płacić na stacjach benzynowych jedynie 2 zł',
            'Prezes Daniel Obajtek ogłosił, że paliwo na stacjach Orlen dla obywateli Ukrainy będzie w stałej cenie – 2 zł/litr',
            'Krążą pogłoski, że Ukraińcy zapłacą mniej za tankowanie, a mowa o obniżce nawet do 2 zł za litr',
            'W sobotę w sieci pojawiły się natomiast wieści o tym, że Ukraińcy z Wyspy Wężowej nie zginęli, tylko zostali wzięci w niewolę.',
            'W mediach społecznościowych pojawiło się nagranie mające pochodzić z momentu, kiedy Rosjanie wysuwali ultimatum wobec ukraińskich obrońców.']

    print('-' * 20)
    for i, s in enumerate(sentences):
        print(f"Sentence {i+1}: {s}")
    print('-' * 20)


    checker = SimilarityChecker()

    print(' ' * 6, end='')
    for i in range(len(sentences)):
        print(f's{i+1}    ', end='')
        print(' ', end='')
    print()

    for i in range(len(sentences)):
        print(f's{i+1}    ', end='')
        for j in range(0, len(sentences)):
            print(f'{checker.similarity(sentences[i], sentences[j]):.3f} ', end='')
            print(' ', end='')
        print()
    print('-' * 20)

    print('Test your own sentences (in Polish):')
    while True:
        print('First sentence:')
        s1 = input()
        print('Second sentence:')
        s2 = input()
        print(f'Similarity: {checker.similarity(s1, s2)}')
        print('-' * 10)
