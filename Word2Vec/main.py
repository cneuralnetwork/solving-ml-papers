import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.spatial.distance import cosine

def process_corpus(file_name):
    with open(file_name, 'r') as f:
        corpus = f.readlines()
    corpus = [sentence for sentence in corpus if sentence.count(" ") >= 2]
    return corpus

class Tokenizer:
    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''):
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
        self.filters = filters
    def fit_on_texts(self, texts):
        for sentence in texts:
            for char in self.filters:
                sentence = sentence.replace(char, ' ')
            words = sentence.lower().split()
            for word in words:
                if word:
                    if word not in self.word_counts:
                        self.word_counts[word] = 1
                    else:
                        self.word_counts[word] += 1
        for i, (word, _) in enumerate(sorted(self.word_counts.items(), 
                                           key=lambda x: x[1], reverse=True), 1):
            self.word_index[word] = i
            self.index_word[i] = word
    def texts_to_sequences(self, texts):
        sequences = []
        for sentence in texts:
            for char in self.filters:
                sentence = sentence.replace(char, ' ')
            words = sentence.lower().split()
            seq = []
            for word in words:
                if word and word in self.word_index:
                    seq.append(self.word_index[word])
            sequences.append(seq)
        return sequences

class SkipgramDataset(Dataset):
    def __init__(self, corpus, window_size, V):
        self.inputs = []
        self.outputs = []
        self.prepare_data(corpus, window_size, V)
    def prepare_data(self, corpus, window_size, V):
        for words in corpus:
            L = len(words)
            for index, word in enumerate(words):
                p = index - window_size
                n = index + window_size + 1
                for i in range(p, n):
                    if i != index and 0 <= i < L:
                        self.inputs.append(word)
                        target = torch.zeros(V)
                        target[words[i]] = 1
                        self.outputs.append(target)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), self.outputs[idx]

class CBOWDataset(Dataset):
    def __init__(self, corpus, window_size, V):
        self.inputs = []
        self.outputs = []
        self.prepare_data(corpus, window_size, V)
    def prepare_data(self, corpus, window_size, V):
        for sentence in corpus:
            L = len(sentence)
            for index, word in enumerate(sentence):
                start = index - window_size
                end = index + window_size + 1
                context_words = []
                for i in range(start, end):
                    if i != index:
                        if 0 <= i < L:
                            context_words.append(sentence[i])
                        else:
                            context_words.append(0)
                self.inputs.append(context_words)
                target = torch.zeros(V)
                target[word] = 1
                self.outputs.append(target)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), self.outputs[idx]

class SkipgramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipgramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)
    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x
    def get_weights(self):
        return [self.embeddings.weight.detach().cpu().numpy()]

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)
    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x
    def get_weights(self):
        return [self.embeddings.weight.detach().cpu().numpy()]

def embed(word, embedding, tokenizer):
    int_word = tokenizer.texts_to_sequences([word])[0]
    one_hot = torch.zeros(len(tokenizer.word_index) + 1)
    one_hot[int_word[0]] = 1
    return one_hot.numpy() @ embedding

def find_similar_words(target_word, embedding, tokenizer, top_n=10):
    if target_word not in tokenizer.word_index:
        return f"Word '{target_word}' not found in vocabulary"
    
    target_word_vector = embed(target_word, embedding, tokenizer)
    all_distances = {}
    
    for word, idx in tokenizer.word_index.items():
        if word == target_word:
            continue
        
        word_vector = np.zeros((len(tokenizer.word_index) + 1))
        word_vector[idx] = 1
        word_vector = word_vector @ embedding
        distance = cosine(target_word_vector, word_vector)
        all_distances[word] = distance
    
    similar_words = sorted(all_distances.items(), key=lambda x: x[1])[:top_n]
    return similar_words


window_size = 2
dims = [50,150,300]
np.random.seed(42)
torch.manual_seed(42)
file_name = 'xxxx'
corpus = process_corpus(file_name)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
corpus = tokenizer.texts_to_sequences(corpus)
V = len(tokenizer.word_index) + 1
skipgram_models = []
for dim in dims:
        model_path=f"skipgram_model_{dim}.pth"
        print(f"Training Skipgram model with {dim} dimensions")
        model = SkipgramModel(V, dim)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = SkipgramDataset(corpus, window_size, V)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        for epoch in range(50):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(dataloader):.4f}")
        skipgram_models.append(model)
        torch.save(model.state_dict(), model_path)
        weights = model.get_weights()
        embedding = weights[0]
        with open(f"vectors_skipgram_{dim}.txt", "w") as f:
            columns = ["word"] + [f"value_{i+1}" for i in range(embedding.shape[1])]
            f.write(" ".join(columns) + "\n")
            for word, i in tokenizer.word_index.items():
                f.write(word + " " + " ".join(map(str, list(embedding[i,:]))) + "\n")
cbow_models = []
for dim in dims:
        model_path=f"cbow_model_{dim}.pth"
        print(f"Training CBOW model with {dim} dimensions")
        model = CBOWModel(V, dim)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = CBOWDataset(corpus, window_size, V)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        for epoch in range(50):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(dataloader):.4f}")
        cbow_models.append(model)
        torch.save(model.state_dict(), model_path)
        weights = model.get_weights()
        embedding = weights[0]
        with open(f"vectors_cbow_{dim}.txt", "w") as f:
            columns = ["word"] + [f"value_{i+1}" for i in range(embedding.shape[1])]
            f.write(" ".join(columns) + "\n")
            for word, i in tokenizer.word_index.items():
                f.write(word + " " + " ".join(map(str, list(embedding[i,:]))) + "\n")
all_models = skipgram_models + cbow_models
model_names = [f"skipgram_{dim}" for dim in dims] + [f"cbow_{dim}" for dim in dims]
embeddings = [model.get_weights()[0] for model in all_models]
    
while True:
        query_word = input("Enter a word to find similar words (or 'exit' to quit): ")
        if query_word.lower() == 'exit':
            break
        
        for model_name, embedding in zip(model_names, embeddings):
            print(f"\nModel: {model_name}")
            similar = find_similar_words(query_word, embedding, tokenizer)
            if isinstance(similar, str):
                print(similar)
            else:
                print(f"Top 10 words similar to '{query_word}':")
                for word, similarity in similar:
                    print(f"{word}: {1-similarity:.4f}")
        print("-" * 40)

