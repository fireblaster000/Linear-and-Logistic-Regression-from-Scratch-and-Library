<h1 align="center">📊 Linear & Logistic Regression Workspace</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.7%2B-blue" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/jupyter-notebook-orange" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/scikit--learn-1.0%2B-green" alt="scikit-learn">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

<p align="center">
This repository contains two main projects:<br>
<b>Multivariate Linear Regression</b> for predicting housing prices.<br>
<b>Logistic Regression Sentiment Analyzer</b> for movie review classification.<br>
Both projects include implementations from scratch and with scikit-learn, and use real-world datasets for hands-on machine learning practice.
</p>

<hr>

<h2>📑 Table of Contents</h2>
<ul>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#datasets">Datasets</a></li>
  <li><a href="#multivariate-linear-regression">Multivariate Linear Regression</a></li>
  <li><a href="#logistic-regression-sentiment-analyzer">Logistic Regression Sentiment Analyzer</a></li>
  <li><a href="#requirements">Requirements</a></li>
  <li><a href="#how-to-run">How to Run</a></li>
  <li><a href="#results--evaluation">Results &amp; Evaluation</a></li>
  <li><a href="#references">References</a></li>
  <li><a href="#author">Author</a></li>
</ul>

<hr>

<h2 id="project-structure">📂 Project Structure</h2>

<pre>
├── Logistic_Reg_Scratch_Lib.ipynb         # Sentiment analysis using logistic regression (scratch & sklearn)
├── Multivariate_Linear_Regression.ipynb   # Housing price prediction using linear regression (scratch & sklearn)
├── LinearReg_HomeDataset/                 # Data files for linear regression
│   ├── trainData.txt
│   ├── trainLabels.txt
│   ├── testData.txt
│   ├── testLabels.txt
├── LogisticReg_Dataset/                   # Data and word lists for sentiment analysis
│   ├── positive_words.txt
│   ├── negative_words.txt
│   ├── stop_words.txt
│   ├── train/
│   │   ├── pos/
│   │   └── neg/
│   └── test/
│       ├── pos/
│       └── neg/
</pre>

<hr>

<h2 id="datasets">📊 Datasets</h2>

<h3>🔹 Linear Regression Dataset</h3>
<ul>
  <li><b>Source:</b> <a href="https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">Boston Housing Dataset</a> (preprocessed)</li>
  <li><b>Files:</b> <code>trainData.txt</code>, <code>trainLabels.txt</code>, <code>testData.txt</code>, <code>testLabels.txt</code></li>
  <li><b>Features:</b> 13 input variables per sample</li>
  <li><b>Target:</b> Housing price</li>
</ul>

<h3>🔹 Logistic Regression Dataset</h3>
<ul>
  <li><b>Source:</b> <a href="https://ai.stanford.edu/~amaas/data/sentiment/">IMDB Large Movie Review Dataset</a></li>
  <li><b>Files:</b> Reviews in <code>train/pos</code>, <code>train/neg</code>, <code>test/pos</code>, <code>test/neg</code></li>
  <li><b>Labels:</b> <code>1</code> (positive), <code>0</code> (negative)</li>
  <li><b>Word Lists:</b> <code>positive_words.txt</code>, <code>negative_words.txt</code>, <code>stop_words.txt</code></li>
</ul>

<hr>

<h2 id="multivariate-linear-regression">🏠 Multivariate Linear Regression</h2>

<p>Implemented in <b>Multivariate_Linear_Regression.ipynb</b></p>

<h3>Steps:</h3>
<ol>
  <li>Data Loading: Reads housing data.</li>
  <li>Feature Scaling: Standardizes features using mean & standard deviation.</li>
  <li>From Scratch: Implements <b>gradient descent</b> for linear regression.</li>
  <li>Scikit-learn: Compares models (<code>LinearRegression</code>, <code>Ridge</code>, <code>Lasso</code>, <code>ElasticNet</code>).</li>
  <li>Visualization: Cost function, MSE vs alpha, weight distributions.</li>
</ol>

<h3>Example Usage:</h3>
<pre><code class="language-python">
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train_scaled, train_labels)

train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

print(f"Train MSE: {mean_squared_error(train_labels, train_preds)}")
print(f"Test MSE: {mean_squared_error(test_labels, test_preds)}")
</code></pre>

<hr>

<h2 id="logistic-regression-sentiment-analyzer">🎬 Logistic Regression Sentiment Analyzer</h2>

<p>Implemented in <b>Logistic_Reg_Scratch_Lib.ipynb</b></p>

<h3>Steps:</h3>
<ol>
  <li>Data Loading: Reads IMDB reviews and labels.</li>
  <li>Text Preprocessing: Lowercasing, removing punctuation, stop-word filtering.</li>
  <li>Feature Extraction: Word-based counts of positive/negative terms.</li>
  <li>From Scratch: Logistic regression with gradient descent, cross-entropy loss, accuracy & F1 evaluation.</li>
  <li>Scikit-learn: Uses <code>LogisticRegression</code> for benchmarking.</li>
  <li>Evaluation: Accuracy, F1 Score, confusion matrix (visualized with heatmaps).</li>
</ol>

<h3>Example Usage:</h3>
<pre><code class="language-python">
model = LogisticRegression(learning_rate=0.001, epochs=5000)
model.fit(np.array(train_Xvals, dtype=float), np.array(train_Yvals, dtype=float),
          np.array(eval_Xvals, dtype=float), np.array(eval_Yvals, dtype=float))

pred_Yvals = model.predict(np.array(test_Xvals))
accuracy, f1, confusion = model.evaluate(test_Yvals, pred_Yvals)

print(f"Accuracy: {accuracy\*100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(confusion)
</code></pre>

<hr>

<h2 id="requirements">⚙️ Requirements</h2>

<ul>
  <li>Python 3.7+</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>matplotlib</li>
  <li>seaborn</li>
  <li>scikit-learn</li>
  <li>tqdm</li>
</ul>

<pre><code class="language-bash">
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
</code></pre>

<hr>

<h2 id="how-to-run">▶️ How to Run</h2>

<ol>
  <li><b>Prepare Datasets</b><br>
  Place dataset files in:
    <ul>
      <li><code>LinearReg_HomeDataset</code></li>
      <li><code>LogisticReg_Dataset</code></li>
    </ul>
  </li>
  <li><b>Open Notebooks</b><br>
    - <code>Multivariate_Linear_Regression.ipynb</code><br>
    - <code>Logistic_Reg_Scratch_Lib.ipynb</code>
  </li>
  <li><b>Run Cells</b><br>
    Execute sequentially to preprocess, train models, and evaluate results.
  </li>
</ol>

<hr>

<h2 id="results--evaluation">📈 Results & Evaluation</h2>

<ul>
  <li><b>Linear Regression:</b> Reports train/test MSE, visualizes cost function & regularization effects.</li>
  <li><b>Logistic Regression:</b> Reports accuracy, F1 score, confusion matrix, and heatmap visualization.</li>
</ul>

<hr>

<h2 id="references">📚 References</h2>

<ol>
  <li><a href="https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">Boston Housing Dataset</a></li>
  <li><a href="https://ai.stanford.edu/~amaas/data/sentiment/">IMDB Large Movie Review Dataset</a></li>
  <li><a href="https://scikit-learn.org/stable/">Scikit-learn Documentation</a></li>
</ol>

<hr>

<h2 id="author">✍️ Author</h2>

<p><b>Ibrahim Farrukh</b></p>
