Homework 1: Multi-Task Learning with a Two-Headed MLP
Course: Deep Learning 
Deadline: 16 November

1. Introduction & Learning Objectives
In this assignment, you will move beyond building standard "single-purpose" networks. Your goal is to design and train a single neural network that can perform two different tasks simultaneously. This is a powerful and efficient technique known as Multi-Task Learning (MTL).
The core idea is that the network can learn a shared, internal representation of the data (the "body") that is useful for multiple problems. This shared knowledge is then fed into separate, task-specific "heads" to get the final predictions.
We will do this using only Fully-Connected (nn.Linear) layers and the optimization/regularization techniques you have already learned.
Learning Objectives:
•	Custom Architecture: Design and implement a custom nn.Module with a shared "body" and two "heads."
•	Tabular Data Preprocessing: Master the pipeline for preparing real-world tabular data (handling categorical and numerical features) for a neural network.
•	Mixed-Task Loss: Implement a custom training loop that combines two different loss functions: MSELoss (for regression) and CrossEntropyLoss (for classification).
•	Multi-Modal Evaluation: Evaluate a single model using different metrics appropriate for each task (MAE for regression, F1-Score for classification).
 
2. The Dataset: UCI Student Performance
We will use the UCI Student Performance Dataset. This dataset contains personal, family, and academic data for students in a Portuguese secondary school.
The data is perfect for our purposes because it contains many input features that could help predict several different outcomes.
•	Download: You only need one of the two files: student-mat.csv (Math) or student-por.csv (Portuguese). student-por.csv is slightly larger and recommended.
Our Tasks:
Your single network will take a student's data as input and predict:
1.	Task 1 (Regression): The student's final grade (G3), a number from 0 to 20.
2.	Task 2 (Classification): Whether the student is in a romantic relationship (romantic), a "yes" or "no" value.
The hypothesis is that a shared "student profile" (learned by the network's body) can help predict both academic performance and personal life.
 
3. Part 1: Data Preprocessing (A Critical Step)
Before you build any model, you must correctly process this data. Do not skip this!
1.	Load Data: Load the .csv file into a pandas DataFrame.
2.	Handle Categorical Features: The dataset has many non-numeric columns (e.g., sex, address, famsup, internet). You must find a robust way to convert these non-numeric values into a numerical format that the network can understand. This includes both binary ('yes'/'no') and multi-class ('Mjob', 'Fjob') features.
3.	Handle Numerical Features: All numerical input features (e.g., age, studytime, failures, G1, G2) should be normalized or standardized. This is a critical step to ensure all features are on a similar scale, which helps with model training.
4.	Create Datasets & Loaders:
o	Separate your data into X (all the processed input features) and two target variables: y_grade (the G3 column) and y_romantic (the binary 0/1 column).
o	Split your data into training, validation, and test sets.
o	Create a custom PyTorch Dataset class that, in its __getitem__ method, returns three items: (x_data, y_grade_data, y_romantic_data).
o	Create your DataLoader for the train, validation, and test sets.
 
4. Part 2: Building the Multi-Head Model
You must now create a new nn.Module that implements the "body-and-heads" architecture. All layers should be nn.Linear.
Your MultiTaskModel class should consist of:
1.	The Shared Body: A sequence of nn.Linear, ReLU, BatchNorm1d, and Dropout layers. This part learns the shared "student profile" from the input features. Feel free to experiment with this architecture.
2.	Head 1: Grade Prediction (Regression): A small MLP that takes the output features from the shared body and outputs 1 single number (the predicted grade).
3.	Head 2: Romantic Status (Classification): A second small MLP that takes the same output features from the shared body and outputs the logits for the 'yes'/'no' classes (e.g., 2 output numbers).
Your forward method must be customized to pass the input x through the body, then send the resulting shared features to both heads, and finally return both predictions.
 
5. Part 3: The Custom Training Loop
This is the most important part of the assignment. You must use two different loss functions and combine them.
Your training loop must be customized to perform the following steps:
1.	Define TWO loss functions: one appropriate for the regression task (e.g., nn.MSELoss()) and one for the classification task (e.g., nn.CrossEntropyLoss()).
2.	Get TWO outputs from your model's forward pass.
3.	Calculate EACH loss separately based on the respective predictions and targets. Be mindful of the data types and shapes required by each loss function.
4.	Combine the losses into a single total_loss (e.g., by summing them).
5.	Perform backpropagation on this total_loss. The AutoGrad system will handle propagating the gradients back through both heads and the shared body.
You must also implement a validation loop that calculates and reports both validation losses.
 
6. Part 4: Evaluation & Analysis
A single "accuracy" score is meaningless here. You must evaluate each task's performance on your test set separately.
For Grade Prediction (Regression):
•	Report the Mean Absolute Error (MAE). This tells us, "On average, how many grade points was our model's prediction off by?" (MAE is better than MSE for interpretation).
For Romantic Status (Classification):
•	Report the Accuracy.
•	Report the F1-Score (for the 'yes' class). This is crucial, especially if the classes are imbalanced, as it balances precision and recall.
Your final report must include these three metrics from the test set.
 
7. What to Submit
Please submit zip file containing Trained Model Weights and Jupyter Notebook that contains:
1.	All your code for data preprocessing, the Dataset class, the MultiTaskModel class, and the training/evaluation loops.
2.	Plots of your training and validation losses over time. You should plot:
o	total_loss (train vs. val)
o	loss_grade (train vs. val)
o	loss_romantic (train vs. val)
3.	A final "Results" section (a markdown cell or printout) that clearly states your final test set performance for MAE, Accuracy, and F1-Score.|
And Saved Model Weights! 
 
8. Bonus Challenge (Optional) (Up to additional 10 points)
The simple summation of losses (total_loss = loss_g + loss_r) works, but it has a potential problem. The MSELoss (e.g., 25.0) might have a much larger scale than the CrossEntropyLoss (e.g., 0.6), meaning the optimizer will focus almost entirely on improving the grade prediction.
A better way is to weight the losses.
Implement a weighted sum of your two losses, controlled by a hyperparameter alpha (a float between 0.0 and 1.0). For example, an alpha of 0.8 would tell the model to "care 80% about the grade and 20% about the romantic status."
Task: Train and evaluate your model at three different alpha values (e.g., 0.2, 0.5, 0.8). Create a table showing how the Test MAE and Test F1-Score change for each alpha. What do you observe?
 
9. Grading & Policies
Point Distribution (Total 100 Points)
•	Part 1: Data Preprocessing (10 points):
•	Part 2: Multi-Head Model (20 points):
•	Part 3: Custom Training Loop (25 points):
•	Part 4: Evaluation & Results (35 points):
•	Code Quality (10 points):
•	Bonus Challenge (10 points):
Policies & Deductions
•	Code Not Running: If your notebook fails to run due to errors, a significant deduction (up to 50%) will be applied.
