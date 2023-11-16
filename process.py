import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from classifier import ToxicityClassifier

# Baixe os recursos necessários do NLTK (neste caso, a lista de palavras stopwords)
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords

def carregar_palavras_toxicas(nome_arquivo):
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            # Use a função set() diretamente no arquivo, dividindo as palavras na nova linha
            return set(arquivo.read().splitlines())
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado.")
        return set()


def tokenizar_frase(frase):
    return word_tokenize(frase.lower())

# Função para preparar os dados de treinamento
def prepare_data(training_data, vocab_to_index):
    data = []
    for sentence, label in training_data:
        tokens = [word.lower() for word in word_tokenize(sentence) if word not in stopwords.words('portuguese')]
        if not tokens:
            continue  # Ignorar frases sem tokens após remoção de stop words
        indices = [vocab_to_index.get(word, vocab_to_index['<UNK>']) for word in tokens]
        data.append((torch.tensor(indices, dtype=torch.long), torch.tensor([label], dtype=torch.float)))

    return data

# ...
def processa_frases(_frases):
    nome_arquivo_palavras = "palavrastoxicas.txt"
    palavras_toxicas = carregar_palavras_toxicas(nome_arquivo_palavras)

    if not palavras_toxicas:
        print("Não foi possível carregar as palavras tóxicas. Verifique o arquivo.")
    else:
        # Exemplo de uso
        vocab_to_index = {'<UNK>': 0}
        vocab_size = 1
        embedding_dim = 50
        output_size = 1
        model = ToxicityClassifier(vocab_size, embedding_dim, output_size)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Lendo frases de um arquivo de texto
        nome_arquivo_training_data = "modelo.txt"
        with open(nome_arquivo_training_data, 'r', encoding='utf-8') as arquivo:
            linhas = arquivo.readlines()

        # Criando training_data a partir das linhas do arquivo
        training_data = []
        for linha in linhas:
            # A linha deve estar no formato "frase, rótulo\n"
            parts = linha.strip().split(',')
            if len(parts) == 2:
                frase, rotulo = parts
                rotulo = int(rotulo)  # Convertendo rótulo para inteiro
                training_data.append((frase, rotulo))
            else:
                print(f"Aviso: A linha '{linha.strip()}' não está no formato esperado e será ignorada.")


        # Adicionando palavras tóxicas diretamente ao array training_data
        for i, (sentence, _) in enumerate(training_data):
            if any(word in palavras_toxicas for word in tokenizar_frase(sentence)):
                training_data[i] = (sentence, 1)

        # Construa o vocabulário com base nos dados de treinamento
        for sentence, _ in training_data:
            tokens = [word.lower() for word in word_tokenize(sentence) if word not in stopwords.words('portuguese')]
            for token in tokens:
                if token not in vocab_to_index:
                    vocab_to_index[token] = vocab_size
                    vocab_size += 1

        # Atualize o tamanho do embedding no modelo
        model.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Treinamento do modelo
        num_epochs = 100
        for epoch in range(num_epochs):
            for sentence, label in training_data:
                model.zero_grad()
                indices = [vocab_to_index.get(word, vocab_to_index['<UNK>']) for word in word_tokenize(sentence.lower())]
                if not indices:
                    continue  # Ignorar frases sem índices após remoção de stop words
                sentence_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1)
                target = torch.tensor([label], dtype=torch.float).view(1, -1)  # Ajuste a forma do tensor de destino

                output = model(sentence_tensor)
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        for frase in _frases:
            indices = [vocab_to_index.get(word, vocab_to_index['<UNK>']) for word in word_tokenize(frase.lower())]
            if not indices:
                continue  # Ignorar frases sem índices após remoção de stop words
            sentence_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1)

            with torch.no_grad():
                model.eval()
                output = model(sentence_tensor)
                predicted_label = "tóxica" if output.item() > 0.5 else "inofensiva"
                outpt_predict = output.item()
            
            target = torch.tensor([1] if any(word in palavras_toxicas for word in tokenizar_frase(frase)) else [0], dtype=torch.float).view(1, -1)  # Ajuste a forma do tensor de destino
            loss = F.binary_cross_entropy(output, target)

            print(f"Frase: '{frase}' - Classificação: {predicted_label} - Predição: {outpt_predict}")
