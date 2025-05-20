# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Design: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import sys
import argparse

class TextBinaryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation (0-255)."""
    val = ord(char)
    return [int(b) for b in format(val, '08b')]

def process_text_file(file_path, chunk_type):
    """Process text file and convert to binary sequences with sliding windows.
    chunk_type: 'unigram' or 'bigram'
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    features = []
    labels = []
    window_size = 784  # Same as MNIST

    if chunk_type == 'unigram':
        # Process as single characters
        binary_data = []
        ascii_chars = []
        print("Converting characters to binary (unigram mode)...")
        for char in tqdm(text, desc="Processing characters"):
            binary_data.extend(char_to_binary(char))
            ascii_chars.append(char)
        # Window slides by 1 character (8 bits)
        if len(binary_data) >= window_size + 8:
            total_windows = len(binary_data) - window_size - 8 + 1
            print("\nCreating sliding windows...")
            for i in tqdm(range(0, total_windows, 8), desc="Creating samples"):
                window_start = i
                window = binary_data[window_start:window_start + window_size]
                next_char_pos = (window_start + window_size) // 8
                if next_char_pos < len(ascii_chars):
                    features.append(window)
                    labels.append(ord(ascii_chars[next_char_pos]) & 127)
        return features, labels

    elif chunk_type == 'bigram':
        # Prepare binary data for all characters
        binary_data = []
        for char in text:
            binary_data.extend(char_to_binary(char))
        # Window slides by 1 character (8 bits)
        if len(binary_data) >= window_size + 16:
            total_windows = len(binary_data) - window_size - 8 + 1  # slide by 8 bits
            print("\nCreating sliding windows...")
            for i in tqdm(range(0, total_windows, 8), desc="Creating samples"):
                window_start = i
                window = binary_data[window_start:window_start + window_size]
                next_char_pos = (window_start + window_size) // 8
                # Prepare label as the next bigram (pad with '\x00' if needed)
                if next_char_pos < len(text):
                    c1 = text[next_char_pos]
                    c2 = text[next_char_pos+1] if (next_char_pos+1) < len(text) else '\x00'
                    label = (ord(c1) << 8) | ord(c2)
                    features.append(window)
                    labels.append(label)
        return features, labels
    else:
        raise ValueError("chunk_type must be 'unigram' or 'bigram'")

def main():
    parser = argparse.ArgumentParser(description="Convert text file to binary dataset with sliding window.")
    parser.add_argument('input_file', help='Input text file')
    parser.add_argument('output_file', help='Output dataset file')
    parser.add_argument('--chunk', required=True, choices=['unigram', 'bigram'], help="Chunking method: 'unigram' (single char) or 'bigram' (pairs, pad with ASCII 0 if needed)")
    args = parser.parse_args()

    # Process the text file
    features, labels = process_text_file(args.input_file, args.chunk)

    # Create the dataset
    dataset = TextBinaryDataset(features, labels)

    # Save the dataset
    with open(args.output_file, 'wb') as f:
        pickle.dump({'features': dataset.features, 'labels': dataset.labels}, f)

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Saved dataset to {args.output_file}")

if __name__ == "__main__":
    main()
