import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_model(model, train_loader, test_loader, device, num_epochs=20, save_path="artifacts/best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    best_accuracy = 0.0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_loss:.4f}")

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with accuracy: {accuracy:.2f}%")

    print(f"\n🏁 Training complete. Best Test Accuracy: {best_accuracy:.2f}%")
    return model
