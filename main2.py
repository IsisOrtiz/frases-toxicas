import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from classifier import ToxicityClassifier
import json
from nltk.corpus import stopwords

def tokenizar_frase(frase):
    return word_tokenize(frase.lower())

def carregar_palavras_toxicas(nome_arquivo):
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            return set(arquivo.read().splitlines())
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado.")
        return set()

def carregar_arquivo():
    try:
        with open("corpus.json", "r", encoding="utf-8") as arquivo:
            dados_json = json.load(arquivo)
        return dados_json
    except FileNotFoundError:
        print(f"Erro: Arquivo corpus.json não encontrado.")
        return []

def processar_frases():
    nome_arquivo_palavras = "palavrastoxicas.txt"
    palavras_toxicas = carregar_palavras_toxicas(nome_arquivo_palavras)

    if not palavras_toxicas:
        print("Não foi possível carregar as palavras tóxicas. Verifique o arquivo.")
    else:
        vocab_to_index = {'<UNK>': 0}
        vocab_size = 1
        embedding_dim = 50
        output_size = 1

        # Mova a definição do modelo e do otimizador para dentro da função
        model = ToxicityClassifier(vocab_size, embedding_dim, output_size)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        linhas = carregar_arquivo()

        training_data = []
        for linha in linhas:
            frase = linha["frase"]
            toxica = linha["toxica"]
            peso = linha["peso"]
            rotulo = 1 if toxica else 0

            for _ in range(1 if peso == 0 else peso):
                training_data.append((frase, rotulo))

        for i, (sentence, _) in enumerate(training_data):
            if any(word in palavras_toxicas for word in tokenizar_frase(sentence)):
                training_data[i] = (sentence, 1)

        for sentence, _ in training_data:
            tokens = [word.lower() for word in word_tokenize(sentence) if word not in stopwords.words('portuguese')]
            for token in tokens:
                if token not in vocab_to_index:
                    vocab_to_index[token] = vocab_size
                    vocab_size += 1

        model.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        num_epochs = 100
        for epoch in range(num_epochs):
            for sentence, label in training_data:
                model.zero_grad()
                indices = [vocab_to_index.get(word, vocab_to_index['<UNK>']) for word in word_tokenize(sentence.lower())]
                if not indices:
                    continue
                sentence_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1)
                target = torch.tensor([label], dtype=torch.float).view(1, -1)

                output = model(sentence_tensor)
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        _frases = [
            "que nojo, falar com você pelo telefone",
            "adoro falar com voce pelo telefone",
            "miha vida seria melhor se vc fizesse seu trabalho",
            "minha vida é melhor com seu trabalho",
            "seu trabalho é bom"
        ]

        for frase in _frases:
            indices = [vocab_to_index.get(word, vocab_to_index['<UNK>']) for word in word_tokenize(frase.lower())]
            if not indices:
                continue
            sentence_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1)

            with torch.no_grad():
                model.eval()
                output = model(sentence_tensor)
                predicted_label = "tóxica" if output.item() > 0.5 else "inofensiva"
                outpt_predict = output.item()

            target = torch.tensor([1] if any(word in palavras_toxicas for word in tokenizar_frase(frase)) else [0], dtype=torch.float).view(1, -1)
            loss = F.binary_cross_entropy(output, target)

            print(f"Frase: '{frase}' - Classificação: {predicted_label} - Predição: {outpt_predict}")

if __name__ == "__main__":
    processar_frases()
