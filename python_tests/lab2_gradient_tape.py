#!/usr/bin/env python3
"""
Lab 2: TensorFlow GradientTape Refresher
Test script for automatic differentiation and custom training loops
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Lab 2: TensorFlow GradientTape Refresher")
    print("=" * 60)
    
    import tensorflow as tf
    
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # 1. Create a trainable variable w
    print("\n1. Creating trainable variable...")
    w = tf.Variable(0.1, dtype=tf.float32, name='weight')
    print(f"Initial w: {w.numpy()}")
    
    # 2. Define inputs and true outputs
    print("\n2. Defining dataset...")
    # Simple linear relationship: y = 3x
    x_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0])
    print(f"x: {x_data.numpy()}")
    print(f"y_true: {y_true.numpy()}")
    
    # 3. Define learning rate
    print("\n3. Setting learning rate...")
    learning_rate = 0.01
    print(f"Learning rate: {learning_rate}")
    
    # 4. Training loop with GradientTape
    print("\n4. Training with GradientTape...")
    num_epochs = 50
    loss_history = []
    
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = w * x_data
            # Compute loss (MSE)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Compute gradient
        grad = tape.gradient(loss, w)
        
        # Update weight
        w.assign_sub(learning_rate * grad)
        
        # Record loss
        loss_history.append(loss.numpy())
        
        # 5. Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: w = {w.numpy():.4f}, loss = {loss.numpy():.6f}")
    
    print(f"\nFinal w: {w.numpy():.4f} (expected: ~3.0)")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    # Plot loss curve
    print("\n6. Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss with GradientTape')
    plt.grid(True)
    plt.savefig('lab2_loss_curve.png')
    print("Loss curve saved to 'lab2_loss_curve.png'")
    plt.close()
    
    # Test predictions
    print("\n7. Testing predictions...")
    test_x = tf.constant([6.0, 7.0, 8.0])
    test_pred = w * test_x
    test_expected = 3.0 * test_x
    
    print("\nPredictions:")
    for x_val, pred_val, exp_val in zip(test_x.numpy(), test_pred.numpy(), test_expected.numpy()):
        print(f"  x={x_val:.1f}: predicted={pred_val:.2f}, expected={exp_val:.2f}")
    
    print("\n" + "=" * 60)
    print("Lab 2 Complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

