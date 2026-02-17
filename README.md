# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1209" height="799" alt="image" src="https://github.com/user-attachments/assets/2a210679-0a23-4590-9a97-84b5380f5907" />

## DESIGN STEPS

### STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

### STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

### STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.

## PROGRAM

### Name: S YASHASWINI
### Register Number: 212224220123

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.model(x)

model = PeopleClassifier(X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
print("Name: yashaswini S")
print("Register Number: 212224220123")
print("\nTraining Output\n")

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

cm = confusion_matrix(y_test, predicted)

print("\n\nConfusion Matrix")
print("Name: yashaswini S")
print("Register Number: 212224220123\n")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report")
print("Name: yashaswini S")
print("Register Number: 212224220123\n")
print(classification_report(y_test, predicted))

# -----------------------------
# New Sample Prediction
# -----------------------------
print("\nNew Sample Data Prediction")
print("Name: yashaswini S")
print("Register Number: 212224220123\n")

# Example sample (change values if needed)
sample = X_test[0].unsqueeze(0)

prediction = model(sample)
_, predicted_class = torch.max(prediction, 1)

segments = ["A", "B", "C", "D"]

print("Input Sample:", sample.numpy())
print("Predicted Segment:", segments[predicted_class.item()])

```

## Dataset Information

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/8d81b690-472d-4ab8-92e7-eded276e1d7e" />

## OUTPUT



### Confusion Matrix

<img width="607" height="553" alt="image" src="https://github.com/user-attachments/assets/256e4ced-04bc-43c7-bdcd-46f75bc830d1" />



### Classification Report

<img width="549" height="269" alt="image" src="https://github.com/user-attachments/assets/a860017d-148d-4dcc-8737-0914fa8c44fa" />



### New Sample Data Prediction

<img width="719" height="249" alt="image" src="https://github.com/user-attachments/assets/02a1d881-b8be-41bb-b750-71e922d9bdac" />


## RESULT
Thus neural network classification model is developded for the given dataset.
