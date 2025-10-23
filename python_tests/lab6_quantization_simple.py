#!/usr/bin/env python3
"""
Lab 6: Model Quantization (Simplified)
Demonstrates quantization to reduce model size and improve inference speed
Note: Simplified for Mac M3 - uses PyTorch's dynamic quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import os

def main():
    print("=" * 60)
    print("Lab 6: Model Quantization (Simplified)")
    print("=" * 60)
    
    # Use CPU for quantization (quantization is best supported on CPU)
    device = "cpu"
    print(f"\nUsing device: {device} (quantization works best on CPU)")
    
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
    
    # Create validation set
    val_texts = [
        (["excellent"], 1), (["awful"], 0), (["great", "good"], 1), (["bad"], 0)
    ] * 10
    val_dataset = SentimentDataset(val_texts)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"✓ Created dataset with {len(train_texts)} training samples")
    
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
    
    model_fp32 = SentimentModel(vocab_size).to(device)
    original_params = sum(p.numel() for p in model_fp32.parameters())
    print(f"✓ Model created with {original_params:,} parameters")
    
    # 3️⃣ Train Model
    print("\n3️⃣ Training baseline model (FP32)...")
    
    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model_fp32.train()
    for epoch in range(20):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model_fp32(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model_fp32.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model_fp32(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1:2d}: Validation Accuracy = {accuracy:.1f}%")
            model_fp32.train()
    
    # Evaluate baseline
    print("\n4️⃣ Evaluating baseline FP32 model...")
    model_fp32.eval()
    
    def evaluate_model(model, dataloader, device):
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        elapsed = time.time() - start_time
        accuracy = 100 * correct / total
        return accuracy, elapsed
    
    fp32_accuracy, fp32_time = evaluate_model(model_fp32, val_loader, device)
    
    # Save model to disk to check size
    torch.save(model_fp32.state_dict(), 'temp_fp32_model.pth')
    fp32_size = os.path.getsize('temp_fp32_model.pth') / 1024  # KB
    
    print(f"  Accuracy: {fp32_accuracy:.1f}%")
    print(f"  Inference time: {fp32_time*1000:.2f} ms")
    print(f"  Model size: {fp32_size:.2f} KB")
    
    # 5️⃣ Simulate Quantization (FP16)
    print("\n5️⃣ Simulating quantization with FP16...")
    print("(Note: Full INT8 quantization requires specific backend support)")
    
    # Create a copy with FP16 precision
    model_fp16 = SentimentModel(vocab_size).to(device)
    model_fp16.load_state_dict(model_fp32.state_dict())
    
    # Convert to FP16
    model_fp16 = model_fp16.half()
    
    print("✓ Model converted to FP16 (half precision)")
    
    # 6️⃣ Evaluate FP16 Model
    print("\n6️⃣ Evaluating FP16 model...")
    model_fp16.eval()
    
    # Need to convert inputs to FP16 for evaluation
    def evaluate_model_fp16(model, dataloader, device):
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Convert back to FP32 for comparison
                outputs = outputs.float()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        elapsed = time.time() - start_time
        accuracy = 100 * correct / total
        return accuracy, elapsed
    
    fp16_accuracy, fp16_time = evaluate_model_fp16(model_fp16, val_loader, device)
    
    # Save FP16 model
    torch.save(model_fp16.state_dict(), 'temp_fp16_model.pth')
    fp16_size = os.path.getsize('temp_fp16_model.pth') / 1024  # KB
    
    print(f"  Accuracy: {fp16_accuracy:.1f}%")
    print(f"  Inference time: {fp16_time*1000:.2f} ms")
    print(f"  Model size: {fp16_size:.2f} KB")
    
    # Clean up temp files
    os.remove('temp_fp32_model.pth')
    os.remove('temp_fp16_model.pth')
    
    # 7️⃣ Compare Results
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print("\n📊 Comparison Table:")
    print(f"{'Metric':<20} | {'FP32':<15} | {'FP16':<15} | {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} | {fp32_accuracy:>13.1f}% | {fp16_accuracy:>13.1f}% | {fp16_accuracy-fp32_accuracy:>+13.1f}%")
    print(f"{'Inference Time':<20} | {fp32_time*1000:>12.2f}ms | {fp16_time*1000:>12.2f}ms | {100*(fp32_time-fp16_time)/fp32_time:>+12.1f}%")
    print(f"{'Model Size':<20} | {fp32_size:>12.2f}KB | {fp16_size:>12.2f}KB | {100*(fp32_size-fp16_size)/fp32_size:>+12.1f}%")
    
    print("\nKey Insights:")
    print("- FP16 quantization reduces model size by ~50%")
    print("- INT8 quantization would reduce by ~75% (with proper backend)")
    print("- Minimal impact on accuracy")
    print("- Faster inference (on appropriate hardware)")
    print("- Lower memory bandwidth requirements")
    
    print("\n" + "=" * 60)
    print("Lab 6 Complete!")
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

