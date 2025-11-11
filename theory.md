# AI & Machine Learning Theory for Oral Preparation

This document covers the theoretical concepts behind the questions in `question-bank.md`, with references to the corresponding lab notebooks (`assi*.ipynb`).

---

### Part 1: Search Algorithms & Classic AI Problems (Q1-Q8)

#### Core Concepts: Search Algorithms

Search algorithms are procedures used to solve problems by systematically exploring a set of possible solutions. In AI, this is often visualized as searching through a **state space graph**.

-   **State Space:** The set of all possible configurations or states that a problem can be in. For a maze, each cell is a state. For a puzzle, each arrangement of pieces is a state.
-   **Node:** A data structure representing a state in the state space.
-   **Start State:** The initial state of the problem (the starting node).
-   **Goal State:** The desired final state or a state that satisfies the goal condition.
-   **Path:** A sequence of states connected by actions.
-   **Solution:** A path from the start state to a goal state.

**Evaluating Search Algorithms:**

-   **Completeness:** Does the algorithm guarantee finding a solution if one exists?
-   **Optimality:** Does the algorithm guarantee finding the best (least-cost) solution?
-   **Time Complexity:** How long does the algorithm take to run? Measured in terms of nodes generated.
-   **Space Complexity:** How much memory does the algorithm need? Measured in terms of the maximum number of nodes stored in memory.

**Complexity Variables:**
-   `b`: **Branching factor** (the maximum number of successors of any node).
-   `d`: **Depth** of the shallowest goal node.
-   `m`: **Maximum depth** of the state space (can be infinite).

**Comparison of Search Algorithms:**

| Algorithm | Type | Completeness | Optimality | Time Complexity | Space Complexity | Key Idea |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BFS** | Uninformed | Yes | Yes (if costs are equal) | O(b^d) | O(b^d) | Explores level by level using a queue. |
| **DFS** | Uninformed | No (in infinite graphs) | No | O(b^m) | O(b*m) | Goes as deep as possible using a stack. |
| **A*** | Informed | Yes | Yes (if heuristic is admissible) | Varies (good heuristic helps) | Varies | Uses a heuristic `h(n)` to guide the search smartly. |

---

#### Q1: Informed vs. Uninformed Search

-   **Uninformed Search (Blind Search):**
    -   **Definition:** These algorithms explore the search space without any information about the problem domain beyond the problem definition. They only know how to traverse nodes and distinguish the goal state. They are "blind" because they have no preference for which node to explore next.
    -   **Examples:** Breadth-First Search (BFS), Depth-First Search (DFS).
    -   **Analogy:** Searching for a room in a building by checking every room in a fixed order (e.g., floor by floor, or going down one hallway to the end before trying another).

-   **Informed Search (Heuristic Search):**
    -   **Definition:** These algorithms use problem-specific knowledge in the form of a **heuristic function**, `h(n)`, to make more intelligent choices.
    -   **Heuristic Function `h(n)`:** An estimate of the cost of the cheapest path from the current state `n` to a goal state. The better the heuristic, the more efficient the search.
    -   **Example:** A* Search.
    -   **Analogy:** Searching for a room in a building, but using signs that point towards the wing where the room is located. You're not just searching blindly; you're using information to guide you.

---

#### Q2: Breadth-First Search (BFS) for Mazes

-   **Concept:** BFS is a graph traversal algorithm that explores all neighbor nodes at the present depth before moving on to the nodes at the next depth level. It uses a **queue** (First-In, First-Out) to manage the nodes to visit. Because it explores layer by layer, it is guaranteed to find the shortest path in terms of the number of edges (or steps) on an unweighted graph.

-   **Algorithm for BFS:**
    1.  Initialize a `queue` and add the `start_node` to it.
    2.  Initialize a `visited` set to keep track of visited nodes and add the `start_node` to it.
    3.  Initialize a way to reconstruct the path (e.g., a dictionary `parent` mapping a node to the node that discovered it).
    4.  **Loop** as long as the `queue` is not empty:
        a.  **Dequeue** the current node (let's call it `current`).
        b.  If `current` is the `goal_node`, the search is complete. Reconstruct the path from the `parent` map and return it.
        c.  For each `neighbor` of `current`:
            i.  If `neighbor` has not been visited:
                -   Add `neighbor` to the `visited` set.
                -   Set `parent[neighbor] = current`.
                -   **Enqueue** `neighbor`.
    5.  If the loop finishes and the goal was not found, return "failure" (no path exists).

---

#### Q3: Depth-First Search (DFS) for Game Maps

-   **Reference:** `assi3.ipynb`
-   **Concept:** DFS explores as far as possible down one branch before backtracking. It uses a **stack** (Last-In, First-Out). This is often implemented implicitly via recursion, which uses the call stack.

-   **Algorithm for Recursive DFS:**
    1.  Define a function `dfs(current_node, goal_node, path, visited)`:
        a.  Mark `current_node` as visited and add it to the `path`.
        b.  If `current_node` is the `goal_node`, return `True` (path found).
        c.  For each `neighbor` of `current_node`:
            i.  If `neighbor` has not been visited:
                -   If `dfs(neighbor, goal_node, path, visited)` returns `True`, then propagate this `True` value up the call stack.
        d.  If the loop finishes without finding the goal, this is a dead end. **Backtrack** by removing `current_node` from the `path`.
        e.  Return `False`.
    2.  To start the search, initialize an empty `path` and an empty `visited` set, then call `dfs(start_node, goal_node, path, visited)`.

-   **Cycle Detection:** The `visited` set is crucial. Before exploring a node, we check if it's in the `visited` set. If it is, we skip it to avoid getting stuck in infinite loops (e.g., Village -> Forest -> Village).

---

#### Q4: A* Search for Pathfinding

-   **Concept:** A* is an informed search algorithm that finds the least-cost path from a start to a goal node. It is both complete and optimal if its heuristic is admissible. It prioritizes nodes that appear to be on the best path to the goal.

-   **Evaluation Function:** `f(n) = g(n) + h(n)`
    -   `n`: The current node.
    -   `g(n)`: The actual cost of the path from the start node to `n`.
    -   `h(n)`: The **heuristic** (estimated cost) from `n` to the goal.
        -   **Admissible Heuristic:** An `h(n)` that **never overestimates** the true cost to the goal. This is required for A* to be optimal. (e.g., Manhattan distance on a grid).
        -   **Consistent Heuristic:** A stronger condition where for any node `n` and its successor `n'`, `h(n) <= cost(n, n') + h(n')`. If a heuristic is consistent, it is also admissible.

-   **Algorithm for A*:**
    1.  Initialize two sets: `open_set` (nodes to be evaluated, often a priority queue) and `closed_set` (nodes already evaluated).
    2.  Add the `start_node` to the `open_set`.
    3.  Set `g(start_node) = 0` and `f(start_node) = h(start_node)`.
    4.  **Loop** as long as `open_set` is not empty:
        a.  Find the node in `open_set` with the lowest `f(n)` score. Let's call it `current`.
        b.  If `current` is the `goal_node`, reconstruct the path and return success.
        c.  Move `current` from `open_set` to `closed_set`.
        d.  For each `neighbor` of `current`:
            i.   If `neighbor` is in `closed_set`, skip it.
            ii.  Calculate a tentative `g_score` for the neighbor: `g(current) + cost(current, neighbor)`.
            iii. If this `g_score` is better than the previously recorded `g(neighbor)`, or if `neighbor` is not in `open_set`:
                -   Record the parent of `neighbor` as `current`.
                -   Set `g(neighbor) = tentative_g_score`.
                -   Set `f(neighbor) = g(neighbor) + h(neighbor)`.
                -   If `neighbor` is not in `open_set`, add it.
    5.  If the loop finishes, no path was found.

---

#### Q5-Q8: Classic AI Problems

These problems are standard benchmarks for demonstrating search algorithms.

-   **Q5: 8-Puzzle Game**
    -   **Problem:** A 3x3 grid with 8 numbered tiles and one blank space. The goal is to arrange the tiles in numerical order.
    -   **Algorithm (A*):**
        1.  **State:** A configuration of the 8 tiles on the grid.
        2.  **Actions:** Moving the blank space Up, Down, Left, or Right.
        3.  **Path Cost `g(n)`:** The number of moves made so far.
        4.  **Heuristic `h(n)`:**
            -   **Misplaced Tiles:** The number of tiles that are not in their goal position. Simple but not very powerful.
            -   **Manhattan Distance:** For each tile, sum the horizontal and vertical distance to its goal position. This is a much better, admissible heuristic.
        5.  The A* algorithm is then used to find the shortest sequence of moves to the solution.

-   **Q6: Tic-Tac-Toe Game**
    -   **Problem:** A two-player game where the goal is to get three of your marks in a row.
    -   **Algorithm (Minimax):**
        1.  **Game Tree:** The algorithm explores the tree of all possible moves.
        2.  **Terminal State:** A state where the game has ended (a player has won, or it's a draw).
        3.  **Utility Function:** A function that assigns a score to a terminal state (e.g., +1 for AI win, -1 for human win, 0 for draw).
        4.  **Logic:**
            -   The **MAX** player (our AI) tries to maximize the score.
            -   The **MIN** player (the opponent) tries to minimize the score.
        5.  The algorithm recursively explores the game tree down to the terminal states. It then propagates the utility values up the tree, with each player choosing the move that leads to the best outcome for them. The AI then makes the move at the root of the tree that leads to the highest possible score, assuming the opponent plays optimally.

-   **Q7: Tower of Hanoi Game**
    -   **Problem:** Move a stack of `n` disks from a source peg to a destination peg, using an auxiliary peg, with the constraint that a larger disk can never be placed on a smaller one.
    -   **Algorithm (Recursive):**
        1.  Define a function `hanoi(n, source, destination, auxiliary)`.
        2.  **Base Case:** If `n == 1`, move the single disk from `source` to `destination`.
        3.  **Recursive Step:**
            a.  Move `n-1` disks from `source` to `auxiliary`, using `destination` as the auxiliary peg: `hanoi(n-1, source, auxiliary, destination)`.
            b.  Move the `nth` (largest) disk from `source` to `destination`.
            c.  Move the `n-1` disks from `auxiliary` to `destination`, using `source` as the auxiliary peg: `hanoi(n-1, auxiliary, destination, source)`.

-   **Q8: Water Jug Problem**
    -   **Problem:** Given two jugs with different capacities and an infinite water supply, find a sequence of operations to obtain a specific amount of water in one of the jugs.
    -   **Algorithm (State Space Search, e.g., BFS):**
        1.  **State:** A pair of numbers `(x, y)` representing the amount of water in each jug.
        2.  **Start State:** `(0, 0)`.
        3.  **Goal State:** Any state where one jug contains the target amount.
        4.  **Actions (State Transitions):**
            -   Fill a jug completely.
            -   Empty a jug.
            -   Pour water from one jug to another until the source is empty or the destination is full.
        5.  Use BFS to explore the state space, as it will find the solution with the minimum number of steps.

---

### Part 2: Data Science & Machine Learning (Q9-Q21)

#### Q9-Q10: EDA and PCA for Prediction

-   **Reference:** `assi9.ipynb`
-   **Exploratory Data Analysis (EDA):**
    -   **Definition:** The process of analyzing and visualizing datasets to summarize their main characteristics, often with visual methods. It is a critical first step before any formal modeling.
    -   **Purpose:** To understand the data, find patterns and relationships, spot anomalies and outliers, check assumptions, and identify important features.
    -   **Techniques in `assi9.ipynb`:**
        -   `df.info()`: Check data types and non-null counts.
        -   `df.describe()`: Get statistical summaries (mean, std, etc.).
        -   `df.isnull().sum()`: Count missing values.
        -   **Visualizations:** Histograms (`sns.histplot`) to see distributions, scatter plots (`sns.scatterplot`) to see relationships between two variables, and box plots (`sns.boxplot`) to see how a numerical variable is distributed across categories.

-   **Principal Component Analysis (PCA):**
    -   **Definition:** A dimensionality reduction technique that transforms a large set of correlated variables into a smaller set of uncorrelated variables called **principal components**. The goal is to retain as much of the original data's variance as possible.
    -   **How it Works (Conceptual Steps):**
        1.  **Standardize the Data:** Scale the data so that each feature has a mean of 0 and a standard deviation of 1. This is crucial because PCA is sensitive to the scale of the features.
        2.  **Covariance Matrix:** Calculate the covariance matrix of the standardized data. This matrix shows how the variables are correlated with each other.
        3.  **Eigen-decomposition:** Compute the **eigenvectors** and **eigenvalues** of the covariance matrix.
            -   **Eigenvectors:** Represent the directions of the new feature space (the principal components). They are orthogonal to each other.
            -   **Eigenvalues:** Represent the magnitude or importance of the corresponding eigenvector. A higher eigenvalue means that its eigenvector explains more of the variance in the data.
        4.  **Select Principal Components:** Rank the eigenvectors by their eigenvalues in descending order. Choose the top `k` eigenvectors to form the new, smaller feature space. The cumulative sum of the explained variance helps decide `k`.
    -   **Application in `assi9.ipynb`:** We reduced the dimensionality of the feature set (from 10 features to 5 principal components) and then trained a Linear Regression model. We compared its performance (R² and RMSE) to a model trained on all original features to see if we could simplify the model with minimal loss in accuracy.

---

#### Q11-Q15: Linear Regression

-   **Reference:** `assi11.ipynb`, `assi12.ipynb`, `assi13.ipynb`, `assi14.ipynb`, `assi15.ipynb`
-   **Concept:** A statistical method for modeling the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.
    -   **Simple Linear Regression:** One feature (`y = w1*x + w0`). (Q12, `assi12.ipynb`)
    -   **Multiple Linear Regression:** Multiple features (`y = w1*x1 + w2*x2 + ... + w0`). (Q11, Q13, Q14, Q15)

-   **Key Components (Implemented from Scratch):**
    -   **Hypothesis `h(x)`:** The linear equation that models the data. For multiple features, it's `h(x) = w^T * x`, where `w` is the vector of weights (coefficients) and `x` is the feature vector (with `x0=1` for the intercept).
    -   **Cost Function (Mean Squared Error - MSE):** Measures the average squared difference between the predicted values and the actual values. `Cost = (1/2n) * Σ(h(x) - y)^2`. The goal is to find the weights `w` that minimize this cost.
    -   **Gradient Descent:** An iterative optimization algorithm used to find the minimum of the cost function. It calculates the gradient (partial derivatives of the cost function with respect to each weight) and takes a step in the opposite direction.
        -   **Update Rule:** `w_j := w_j - α * (1/n) * Σ(h(x) - y) * x_j`
        -   `α` (alpha) is the **learning rate**, which controls the size of each step.
    -   **R-squared (R²):** A metric from 0 to 1 that indicates the proportion of the variance in the target variable that is predictable from the features. A score of 1.0 is a perfect fit.

-   **K-Fold Cross-Validation:**
    -   **Definition:** A resampling procedure used to evaluate a model on a limited data sample, providing a more robust estimate of its performance on unseen data.
    -   **Process:**
        1.  Shuffle the dataset randomly.
        2.  Split the dataset into `k` groups (folds).
        3.  For each fold:
            a.  Take the fold as a hold-out or **test set**.
            b.  Take the remaining `k-1` folds as a **training set**.
            c.  Train the model on the training set and evaluate it on the test set.
        4.  Average the evaluation scores from all `k` folds to get the final performance metric.
    -   **Application:** Used in `assi11`, `assi13`, `assi14`, and `assi15` to validate the regression models.

-   **Differences in Regression Assignments:**
    -   **Q12 (`assi12.ipynb`):** From-scratch simple linear regression (one feature). Focuses on the core mechanics.
    -   **Q11, Q13, Q14 (`assi11.ipynb`, `assi13.ipynb`, `assi14.ipynb`):** Multiple linear regression with K-Fold CV on different datasets. The core logic is the same, but the features and data preprocessing (like encoding categorical variables in `assi14.ipynb`) differ.
    -   **Q15 (`assi15.ipynb`):** Uses `TimeSeriesSplit` for cross-validation, which is crucial for time-series data. It ensures that the test set always comes after the training set, preventing the model from "looking into the future."

---

#### Q16-Q17 & Q18-Q21: Classification Algorithms

-   **Reference:** `assi16.ipynb` (Naïve Bayes), `assi19.ipynb`, `assi20.ipynb`, `assi21.ipynb` (SVM)

**Comparison of Classification Algorithms:**

| Algorithm | Key Idea | Pros | Cons | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **Naïve Bayes** | Uses Bayes' theorem with a "naïve" assumption of feature independence. | Fast, simple, works well with high dimensions, good for text. | The independence assumption is often violated in reality. | Text classification (spam, sentiment), real-time prediction. |
| **SVM** | Finds the hyperplane that best separates classes by maximizing the margin. | Effective in high-dimensional spaces, memory efficient, versatile with kernels. | Can be slow on large datasets, sensitive to kernel choice and parameters. | Image classification, bioinformatics, when a clear margin of separation is expected. |

---

#### Q16-Q17: Naïve Bayes

-   **Concept:** A probabilistic classifier based on **Bayes' Theorem**.
    -   `P(A|B) = (P(B|A) * P(A)) / P(B)`
    -   In classification: `P(Class|Features) = (P(Features|Class) * P(Class)) / P(Features)`
-   **The "Naïve" Assumption:** It assumes that all features are **conditionally independent** of one another, given the class. This is a strong (or "naïve") assumption but simplifies the math (`P(Features|Class)` becomes the product of `P(feature_i|Class)` for all features) and works surprisingly well in practice.
-   **Application (`assi16.ipynb`):** For email spam detection, the algorithm calculates the probability of an email being spam given the words it contains. It compares `P(Spam | words)` with `P(Not Spam | words)` and picks the class with the higher probability.

-   **Evaluation Metrics for Classification:**
    -   **Confusion Matrix:** A table showing the performance of a classification model.
        -   **True Positives (TP):** Correctly predicted positive (e.g., spam email correctly identified as spam).
        -   **True Negatives (TN):** Correctly predicted negative (e.g., normal email correctly identified as normal).
        -   **False Positives (FP):** Incorrectly predicted positive (Type I Error, e.g., normal email flagged as spam).
        -   **False Negatives (FN):** Incorrectly predicted negative (Type II Error, e.g., spam email missed and sent to inbox).
    -   **Accuracy:** `(TP + TN) / Total`. Overall correctness. Can be misleading on imbalanced datasets.
    -   **Precision:** `TP / (TP + FP)`. Of all positive predictions, how many were correct? (Measures the cost of false positives).
    -   **Recall (Sensitivity):** `TP / (TP + FN)`. Of all actual positives, how many did we find? (Measures the cost of false negatives).
    -   **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)`. The harmonic mean of Precision and Recall. A good balanced measure.

---

#### Q18-Q21: Support Vector Machine (SVM)

-   **Concept:** A supervised learning model that finds the optimal **hyperplane** (a decision boundary) that separates data points of different classes with the maximum possible **margin**.
-   **Hyperplane:** In a 2D space, it's a line. In a 3D space, it's a plane. In higher dimensions, it's a hyperplane.
-   **Margin:** The distance between the hyperplane and the nearest data point from either class. SVM aims to maximize this margin.
-   **Support Vectors:** The data points that lie closest to the hyperplane and define the margin. If these points were moved, the hyperplane would move too.

-   **The Kernel Trick:** For data that is not linearly separable, SVM uses **kernels** to project the data into a higher-dimensional space where it *can* be separated by a hyperplane. This is done computationally efficiently without actually creating the new dimensions.
    -   **Linear Kernel:** For linearly separable data. No transformation is applied.
    -   **Polynomial Kernel (Q20, Q21):** `(gamma * x^T * y + r)^d`. Creates non-linear, curved decision boundaries. Used in `assi20.ipynb` and `assi21.ipynb`.
    -   **RBF Kernel (Radial Basis Function):** Another popular non-linear kernel, which can handle even more complex relationships.

-   **Differences in SVM Assignments:**
    -   **Q18 & Q19 (`assi19.ipynb`):** SVM for spam detection. The "from scratch" version focuses on implementing the gradient descent logic to find the optimal weights for the hyperplane.
    -   **Q20 (`assi20.ipynb`):** Uses a Polynomial Kernel to capture more complex relationships in the student performance dataset.
    -   **Q21 (`assi21.ipynb`):** Also uses a Polynomial Kernel for the breast cancer dataset. This question emphasizes evaluation with a **ROC Curve**.

-   **ROC Curve (Receiver Operating Characteristic):**
    -   **Definition:** A plot that shows the performance of a classification model at all classification thresholds. It plots the **True Positive Rate (Recall)** against the **False Positive Rate** (`FP / (FP + TN)`).
    -   **How it works:** Most classifiers output a probability or score. By varying the threshold for what is considered a "positive" prediction (e.g., >0.5, >0.6, >0.7), we get different pairs of TPR and FPR, which trace out the curve.
    -   **AUC (Area Under the Curve):** The area under the ROC curve. A model with an AUC of 1.0 is perfect, while an AUC of 0.5 is no better than random guessing. It provides a single number to summarize the model's performance across all thresholds.
