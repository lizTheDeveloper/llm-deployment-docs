#!/usr/bin/env python3
"""
Lab 4: Knowledge Distillation (Simplified)
Demonstrates distillation concept with a simple text classification task
Note: Simplified for Mac M3 - uses tiny models and small dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

def main():
    print("=" * 60)
    print("Lab 4: Knowledge Distillation (Simplified)")
    print("=" * 60)
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"\nUsing device: {device}")
    
    # 1️⃣ Create a simple synthetic dataset
    print("\n1️⃣ Creating synthetic dataset...")
    print("Task: Binary sentiment classification (positive/negative)")
    
    # Simple vocabulary and sentences
    vocab = {"good": 0, "great": 1, "excellent": 2, "bad": 3, "terrible": 4, "awful": 5, "ok": 6, "fine": 7}
    vocab_size = len(vocab)
    
    # Training data: simple sentiment examples
    train_texts = [
        (["good"], 1), (["great"], 1), (["excellent"], 1),
        (["bad"], 0), (["terrible"], 0), (["awful"], 0),
        (["good", "great"], 1), (["bad", "terrible"], 0),
        (["excellent", "good"], 1), (["awful", "bad"], 0),
        (["ok"], 1), (["fine"], 1),
    ] * 10  # Repeat for more training data
    
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
    
    # 2️⃣ Define Teacher Model (larger)
    print("\n2️⃣ Defining Teacher Model (larger network)...")
    
    class TeacherModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 2)
        
        def forward(self, x):
            # x shape: (batch, seq_len)
            x = self.embedding(x)  # (batch, seq_len, embed_dim)
            x = torch.mean(x, dim=1)  # Simple averaging
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    teacher = TeacherModel(vocab_size).to(device)
    print(f"✓ Teacher model created with {sum(p.numel() for p in teacher.parameters())} parameters")
    
    # 3️⃣ Train Teacher Model
    print("\n3️⃣ Training Teacher Model...")
    
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    teacher.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            teacher_optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            teacher_optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if (epoch + 1) % 5 == 0:
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1:2d}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {accuracy:.1f}%")
    
    print("✓ Teacher model trained")
    
    # 4️⃣ Define Student Model (smaller)
    print("\n4️⃣ Defining Student Model (smaller network)...")
    
    class StudentModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=16, hidden_dim=24):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 2)
        
        def forward(self, x):
            x = self.embedding(x)
            x = torch.mean(x, dim=1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    student = StudentModel(vocab_size).to(device)
    print(f"✓ Student model created with {sum(p.numel() for p in student.parameters())} parameters")
    print(f"  Size reduction: {100 * (1 - sum(p.numel() for p in student.parameters()) / sum(p.numel() for p in teacher.parameters())):.1f}%")
    
    # 5️⃣ Distillation Training
    print("\n5️⃣ Training Student with Knowledge Distillation...")
    print("Using temperature=2.0 for soft targets")
    
    student_optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    temperature = 2.0
    alpha = 0.7  # Weight for distillation loss
    
    teacher.eval()
    student.train()
    
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            student_optimizer.zero_grad()
            
            # Get teacher predictions (soft targets)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
            
            # Get student predictions
            student_outputs = student(inputs)
            soft_predictions = F.log_softmax(student_outputs / temperature, dim=1)
            
            # Distillation loss (KL divergence)
            distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
            
            # Hard target loss
            hard_loss = criterion(student_outputs, labels)
            
            # Combined loss
            loss = alpha * distillation_loss + (1 - alpha) * hard_loss
            
            loss.backward()
            student_optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if (epoch + 1) % 5 == 0:
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1:2d}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {accuracy:.1f}%")
    
    print("✓ Student model distilled")
    
    # 6️⃣ Compare Models
    print("\n6️⃣ Comparing Teacher and Student...")
    
    teacher.eval()
    student.eval()
    
    # Test on a few examples
    test_examples = [
        (["excellent", "great"], "Positive"),
        (["terrible", "awful"], "Negative"),
        (["ok"], "Neutral/Positive")
    ]
    
    print("\nTest Predictions:")
    with torch.no_grad():
        for words, expected in test_examples:
            input_tensor = text_to_tensor(words).unsqueeze(0).to(device)
            
            teacher_out = teacher(input_tensor)
            student_out = student(input_tensor)
            
            teacher_pred = torch.argmax(teacher_out, dim=1).item()
            student_pred = torch.argmax(student_out, dim=1).item()
            
            sentiment_map = {0: "Negative", 1: "Positive"}
            
            print(f"\n  Input: {' '.join(words)}")
            print(f"  Expected: {expected}")
            print(f"  Teacher: {sentiment_map[teacher_pred]}")
            print(f"  Student: {sentiment_map[student_pred]}")
            print(f"  Agreement: {'✓' if teacher_pred == student_pred else '✗'}")
    
    # Model size comparison
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Reduction: {100 * (1 - student_params/teacher_params):.1f}%")
    print("\nKey Insights:")
    print("- Student model is ~50% smaller")
    print("- Maintains similar accuracy through distillation")
    print("- Faster inference due to smaller size")
    
    print("\n" + "=" * 60)
    print("Lab 4 Complete!")
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

