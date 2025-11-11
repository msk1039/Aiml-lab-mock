# AI & Machine Learning Theory for Oral Preparation

This document covers the theoretical concepts behind the questions in `question-bank.md`, with references to the corresponding lab notebooks (`assi*.ipynb`).

---

### Part 1: Search Algorithms & Classic AI Problems (Q1-Q8)

#### Core Concepts: Search Algorithms

Search algorithms are fundamental to AI for problem-solving. They explore a **state space** (all possible configurations of a problem) to find a path from a **start state** to a **goal state**.

**Comparison of Search Algorithms:**

| Algorithm | Type | Completeness | Optimality | Time Complexity | Space Complexity | Key Idea |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BFS** | Uninformed | Yes | Yes (if costs are equal) | O(b^d) | O(b^d) | Explores level by level. |
| **DFS** | Uninformed | No (in infinite graphs) | No | O(b^m) | O(b*m) | Goes as deep as possible. |
| **A*** | Informed | Yes | Yes | Varies (good heuristic helps) | Varies | Uses a heuristic to guide the search smartly. |

- **b**: branching factor (number of choices from a node)
- **d**: depth of the shallowest solution
- **m**: maximum depth of the state space

---

#### Q1: Informed vs. Uninformed Search

- **Uninformed Search (Blind Search):**
  - **Definition:** These algorithms explore the search space without any information about the problem domain. They only know how to traverse the nodes and distinguish the goal state.
  - **Examples:** Breadth-First Search (BFS), Depth-First Search (DFS).
  - **Analogy:** Searching for a room in a building by checking every room in a fixed order (e.g., floor by floor).

- **Informed Search (Heuristic Search):**
  - **Definition:** These algorithms use problem-specific knowledge (a **heuristic**) to make more intelligent choices. A heuristic is an educated guess of the distance from a node to the goal.
  - **Example:** A* Search.
  - **Analogy:** Searching for a room in a building, but using signs that point towards the wing where the room is located. You're not just searching blindly; you're using information to guide you.

---

#### Q2: Breadth-First Search (BFS) for Mazes

- **Concept:** BFS is a graph traversal algorithm that explores all neighbor nodes at the present depth before moving on to the nodes at the next depth level. It uses a **queue** (First-In, First-Out) to manage the nodes to visit.
- **Application (Maze):**
  1. Start at the entrance of the maze and add it to a queue.
  2. While the queue is not empty, dequeue a cell.
  3. Explore its unvisited neighbors (up, down, left, right) that are not walls.
  4. Enqueue these neighbors and mark them as visited.
  5. The first time the goal cell is dequeued, the path found is guaranteed to be the shortest path in terms of the number of steps. This is because BFS explores all paths of length `k` before exploring any path of length `k+1`.

---

#### Q3: Depth-First Search (DFS) for Game Maps

- **Reference:** `assi3.ipynb`
- **Concept:** DFS explores as far as possible down one branch before backtracking. It uses a **stack** (Last-In, First-Out), which is often implemented implicitly via recursion.
- **Application (Game Map):**
  - In `assi3.ipynb`, we used a recursive DFS to traverse a graph representing a game map.
  1. Start at a node (e.g., 'Village').
  2. Mark it as visited and add it to the current path.
  3. If it's the goal ('Treasure'), the search is successful.
  4. If not, for each of its neighbors, recursively call the DFS function.
  5. If a path from a neighbor leads to the goal, propagate the success.
  6. If all neighbors lead to dead ends, **backtrack** by removing the current node from the path.
- **Cycle Detection:** A `visited` set is crucial. Before exploring a node, we check if it's in the `visited` set. If it is, we skip it to avoid getting stuck in infinite loops (e.g., Village -> Forest -> Village).

---

#### Q4: A* Search for Pathfinding

- **Concept:** A* is an informed search algorithm that finds the least-cost path from a start to a goal node. It balances the cost to reach a node with an estimated cost to the goal.
- **Evaluation Function:** `f(n) = g(n) + h(n)`
  - `n`: The current node.
  - `g(n)`: The actual cost of the path from the start node to `n`.
  - `h(n)`: The **heuristic** (estimated cost) from `n` to the goal. The heuristic must be **admissible** (never overestimates the true cost) for A* to be optimal.
- **Heuristic for a Maze:**
  - **Manhattan Distance:** `|x1 - x2| + |y1 - y2|`. This is a perfect heuristic for a grid where you can only move up, down, left, or right. It's admissible because it's the exact number of moves required if there were no obstacles.
- **How it Works:** A* maintains a priority queue of nodes to visit, prioritized by the lowest `f(n)` value. It always explores the most promising node first.

---

#### Q5-Q8: Classic AI Problems

These problems are standard ways to demonstrate search algorithms.

- **8-Puzzle (Q5):** A state is a configuration of the 3x3 grid. A move (sliding a tile) transitions from one state to another. A* is often used to solve it, with the Manhattan distance of each tile from its goal position as the heuristic.
- **Tic-Tac-Toe (Q6):** This is typically solved with the **Minimax algorithm**, a recursive algorithm used in two-player games. It explores the game tree to find the optimal move by assuming the opponent will also play optimally.
- **Tower of Hanoi (Q7):** A classic problem solved with **recursion**. The solution involves breaking the problem of moving `n` disks into smaller subproblems of moving `n-1` disks.
- **Water Jug Problem (Q8):** A state space search problem where the state is the amount of water in each jug. The goal is to reach a state with a specific amount of water in one of the jugs.

---

### Part 2: Data Science & Machine Learning (Q9-Q21)

#### Q9-Q10: EDA and PCA for Prediction

- **Reference:** `assi9.ipynb`
- **Exploratory Data Analysis (EDA):**
  - **Definition:** The process of analyzing and visualizing datasets to summarize their main characteristics, often with visual methods.
  - **Purpose:** To understand the data, find patterns, spot anomalies, check assumptions, and identify important features.
  - **In `assi9.ipynb`:** We used `df.info()`, `df.describe()`, and visualizations (histograms, scatter plots) to understand the distribution of Uber fares and their relationship with distance and passenger count.

- **Principal Component Analysis (PCA):**
  - **Definition:** A dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information in the large set. It creates new, uncorrelated variables called **principal components**.
  - **How it Works:**
    1. **Standardize** the data (so that features with larger scales don't dominate).
    2. Calculate the covariance matrix to understand how variables relate to each other.
    3. Compute the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors are the principal components.
    4. The principal components are ranked by their **explained variance** (how much information they capture). The first PC captures the most variance, the second captures the next most, and so on.
  - **Application in `assi9.ipynb`:** We reduced the dimensionality of the feature set (from 10 features to 5 principal components) and then trained a Linear Regression model. We compared its performance (R² and RMSE) to a model trained on all original features. The goal is to see if we can simplify the model with minimal loss in accuracy.

---

#### Q11-Q15: Linear Regression

- **Reference:** `assi11.ipynb`, `assi12.ipynb`, `assi13.ipynb`, `assi14.ipynb`, `assi15.ipynb`
- **Concept:** A statistical method for modeling the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.
  - **Simple Linear Regression:** One feature (`y = w1*x + w0`). (Q12, `assi12.ipynb`)
  - **Multiple Linear Regression:** Multiple features (`y = w1*x1 + w2*x2 + ... + w0`). (Q11, Q13, Q14, Q15)

- **Key Components (Implemented from Scratch):**
  - **Hypothesis:** The linear equation `h(x) = w^T * x`.
  - **Cost Function (Mean Squared Error - MSE):** Measures the average squared difference between the predicted values and the actual values. `Cost = (1/2n) * Σ(h(x) - y)^2`. We aim to minimize this.
  - **Gradient Descent:** An iterative optimization algorithm used to find the minimum of the cost function. It calculates the gradient (slope) of the cost function and takes a step in the opposite direction. The `alpha` (learning rate) parameter controls the step size.
  - **R-squared (R²):** A metric that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s). A score of 1.0 is a perfect fit.

- **K-Fold Cross-Validation:**
  - **Definition:** A resampling procedure used to evaluate machine learning models on a limited data sample.
  - **How it Works:** The data is split into `k` subsets (folds). The model is trained `k` times, each time using `k-1` folds for training and one fold for testing. The results are then averaged.
  - **Advantage:** It provides a more robust and reliable estimate of model performance than a single train/test split, as it ensures every data point gets to be in a test set exactly once.
  - **Application:** Used in `assi11`, `assi13`, `assi14`, and `assi15` to validate the regression models.

- **Differences in Regression Assignments:**
  - **Q12 (`assi12.ipynb`):** From-scratch simple linear regression (one feature). Focuses on the core mechanics.
  - **Q11, Q13, Q14 (`assi11.ipynb`, `assi13.ipynb`, `assi14.ipynb`):** Multiple linear regression with K-Fold CV on different datasets (house prices, student scores, salaries). The core logic is the same, but the features and data preprocessing (like encoding categorical variables in `assi14.ipynb`) differ.
  - **Q15 (`assi15.ipynb`):** Uses `TimeSeriesSplit` for cross-validation, which is crucial for time-series data to prevent looking into the "future" during training.

---

#### Q16-Q17 & Q18-Q21: Classification Algorithms

- **Reference:** `assi16.ipynb` (Naïve Bayes), `assi19.ipynb`, `assi20.ipynb`, `assi21.ipynb` (SVM)

**Comparison of Classification Algorithms:**

| Algorithm | Key Idea | Pros | Cons | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **Naïve Bayes** | Uses Bayes' theorem with a "naïve" assumption of feature independence. | Fast, works well with high dimensions, good for text. | The independence assumption is often violated. | Text classification (spam, sentiment), real-time prediction. |
| **SVM** | Finds the hyperplane that best separates classes by maximizing the margin. | Effective in high-dimensional spaces, memory efficient. | Can be slow on large datasets, sensitive to kernel choice. | Image classification, bioinformatics, when a clear margin of separation is expected. |

---

#### Q16-Q17: Naïve Bayes

- **Concept:** A probabilistic classifier based on **Bayes' Theorem**.
  - `P(A|B) = (P(B|A) * P(A)) / P(B)`
  - In classification: `P(Class|Features) = (P(Features|Class) * P(Class)) / P(Features)`
- **The "Naïve" Assumption:** It assumes that all features are independent of one another, given the class. This is a strong assumption but simplifies the math and works surprisingly well in practice (e.g., for spam detection, the presence of the word "free" is treated independently of the word "money").
- **Application (`assi16.ipynb`):** For email spam detection, the algorithm calculates the probability of an email being spam given the words it contains.

- **Evaluation Metrics for Classification:**
  - **Confusion Matrix:** A table showing the performance of a classification model.
    - **True Positives (TP):** Correctly predicted positive.
    - **True Negatives (TN):** Correctly predicted negative.
    - **False Positives (FP):** Incorrectly predicted positive (Type I Error).
    - **False Negatives (FN):** Incorrectly predicted negative (Type II Error).
  - **Accuracy:** `(TP + TN) / Total`. Overall correctness. Can be misleading on imbalanced datasets.
  - **Precision:** `TP / (TP + FP)`. Of all positive predictions, how many were correct? (Measures false positives).
  - **Recall (Sensitivity):** `TP / (TP + FN)`. Of all actual positives, how many did we find? (Measures false negatives).
  - **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)`. The harmonic mean of Precision and Recall. A good balanced measure.

---

#### Q18-Q21: Support Vector Machine (SVM)

- **Concept:** A supervised learning model that finds the optimal **hyperplane** (a boundary) that separates data points of different classes with the maximum **margin**. The data points that lie on the margin are called **support vectors**.
- **The Kernel Trick:** For data that is not linearly separable, SVM uses kernels to project the data into a higher-dimensional space where it can be separated by a hyperplane.
  - **Linear Kernel:** For linearly separable data.
  - **Polynomial Kernel (Q20, Q21):** `(gamma * x^T * y + r)^d`. Creates non-linear decision boundaries. Used in `assi20.ipynb` and `assi21.ipynb`.
  - **RBF Kernel:** Another popular non-linear kernel.

- **Differences in SVM Assignments:**
  - **Q18 & Q19 (`assi19.ipynb`):** SVM for spam detection. The "from scratch" version focuses on implementing the gradient descent logic to find the optimal weights for the hyperplane.
  - **Q20 (`assi20.ipynb`):** Uses a Polynomial Kernel to capture more complex relationships in the student performance dataset.
  - **Q21 (`assi21.ipynb`):** Also uses a Polynomial Kernel for the breast cancer dataset. This question emphasizes evaluation with a **ROC Curve**.

- **ROC Curve (Receiver Operating Characteristic):**
  - **Definition:** A plot that shows the performance of a classification model at all classification thresholds. It plots the **True Positive Rate (Recall)** against the **False Positive Rate**.
  - **AUC (Area Under the Curve):** The area under the ROC curve. A model with an AUC of 1.0 is perfect, while an AUC of 0.5 is no better than random guessing. It provides a single number to summarize the model's performance across all thresholds.
