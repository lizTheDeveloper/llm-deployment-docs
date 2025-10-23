#!/usr/bin/env python3
"""
Lab 1: Keras Quick Refresher
Test script for basic Keras operations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Lab 1: Keras Quick Refresher")
    print("=" * 60)
    
    # Import TensorFlow and Keras
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # 1. Generate a small synthetic dataset
    print("\n1. Generating synthetic dataset...")
    np.random.seed(42)
    num_samples = 100
    x_data = np.linspace(0, 10, num_samples)
    y_data = 3 * x_data + 7 + np.random.randn(num_samples) * 2  # y = 3x + 7 + noise
    
    # Reshape for Keras
    x_train = x_data.reshape(-1, 1)
    y_train = y_data.reshape(-1, 1)
    
    print(f"Generated {num_samples} samples")
    print(f"X shape: {x_train.shape}, Y shape: {y_train.shape}")
    
    # 2. Define a small Keras model
    print("\n2. Defining Keras model...")
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])
    
    print("Model architecture:")
    model.summary()
    
    # 3. Compile the model
    print("\n3. Compiling model...")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Model compiled with optimizer='adam' and loss='mse'")
    
    # 4. Train the model
    print("\n4. Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    final_loss = history.history['loss'][-1]
    print(f"\nFinal training loss: {final_loss:.4f}")
    
    # 5. Plot the loss curve
    print("\n5. Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('lab1_loss_curve.png')
    print("Loss curve saved to 'lab1_loss_curve.png'")
    plt.close()
    
    # 6. Test the model on new values
    print("\n6. Testing model predictions...")
    test_x = np.array([[2.0], [5.0], [8.0]])
    predictions = model.predict(test_x, verbose=0)
    
    print("\nPredictions:")
    for x_val, pred_val in zip(test_x.flatten(), predictions.flatten()):
        expected = 3 * x_val + 7
        print(f"  x={x_val:.1f}: predicted={pred_val:.2f}, expected={expected:.2f}")
    
    print("\n" + "=" * 60)
    print("Lab 1 Complete!")
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

