#!/usr/bin/env python3
"""
Lab 5: Model Pruning (Simplified)
Demonstrates pruning techniques to reduce model size
Note: Simplified for Mac M3 - uses a simple model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
import numpy as np

def main():
    print("=" * 60)
    print("Lab 5: Model Pruning (Simplified)")
    print("=" * 60)
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"\nUsing device: {device}")
    
    # 1️⃣ Create simple dataset
    print("\n1️⃣ Creating synthetic dataset...")
    
    vocab = {"good": 0, "great": 1, "excellent": 2, "bad": 3, "terrible": 4, "awful": 5, "ok": 6, "fine": 7}
    vocab_size = len(vocab)
    
    train_texts = [
        (["good"], 1), (["great"], 1), (["excellent"], 1),
        (["bad"], 0), (["terrible"], 0), (["awful"], 0),
        (["good", "great"], 1), (["bad", "terrible"], 0),
    ] * 15
    
    def text_to_tensor(words, max_len=5):
        indices = [vocab.get(w, 0) for w in words]
        indices = indices[:max_len] + [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    
    class SentimentDataset(Dataset):
        def __init__(self, texts):
            self.data = texts
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            words, label = self.data[idx]
            return text_to_tensor(words), torch.tensor(label, dtype=torch.long)
    
    train_dataset = SentimentDataset(train_texts)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    print(f"✓ Created dataset with {len(train_texts)} samples")
    
    # 2️⃣ Define Model
    print("\n2️⃣ Defining model...")
    
    class SentimentModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 2)
        
        def forward(self, x):
            x = self.embedding(x)
            x = torch.mean(x, dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    model = SentimentModel(vocab_size).to(device)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {original_params:,} parameters")
    
    # 3️⃣ Train Model
    print("\n3️⃣ Training baseline model...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if (epoch + 1) % 5 == 0:
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1:2d}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {accuracy:.1f}%")
    
    baseline_accuracy = 100 * correct / total
    print(f"✓ Baseline accuracy: {baseline_accuracy:.1f}%")
    
    # 4️⃣ Apply Pruning
    print("\n4️⃣ Applying unstructured pruning...")
    
    def compute_sparsity(model):
        """Compute the sparsity (percentage of zero weights) in the model"""
        total_params = 0
        zero_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    # Already pruned
                    total_params += module.weight_mask.numel()
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
        
        return 100.0 * zero_params / total_params if total_params > 0 else 0
    
    print(f"Initial sparsity: {compute_sparsity(model):.2f}%")
    
    # Apply L1 unstructured pruning to linear layers
    pruning_amount = 0.3  # Prune 30% of weights
    print(f"\nPruning {pruning_amount*100:.0f}% of weights in each linear layer...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            print(f"  Pruned layer: {name}")
    
    print(f"\nSparsity after pruning: {compute_sparsity(model):.2f}%")
    
    # 5️⃣ Fine-tune Pruned Model
    print("\n5️⃣ Fine-tuning pruned model...")
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if (epoch + 1) % 5 == 0:
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1:2d}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {accuracy:.1f}%")
    
    pruned_accuracy = 100 * correct / total
    print(f"✓ Pruned model accuracy: {pruned_accuracy:.1f}%")
    
    # 6️⃣ Make pruning permanent
    print("\n6️⃣ Making pruning permanent...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    print("✓ Pruning masks removed (weights are now permanently zero)")
    
    # 7️⃣ Compare results
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"Original parameters: {original_params:,}")
    print(f"Sparsity achieved: {compute_sparsity(model):.2f}%")
    print(f"\nAccuracy comparison:")
    print(f"  Baseline:      {baseline_accuracy:.1f}%")
    print(f"  After pruning: {pruned_accuracy:.1f}%")
    print(f"  Difference:    {pruned_accuracy - baseline_accuracy:+.1f}%")
    
    print("\nKey Insights:")
    print(f"- Removed {pruning_amount*100:.0f}% of weights from each linear layer")
    print("- Model maintains similar accuracy")
    print("- Sparse weights can be stored more efficiently")
    print("- Inference can be faster with sparse matrix operations")
    
    print("\n" + "=" * 60)
    print("Lab 5 Complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

