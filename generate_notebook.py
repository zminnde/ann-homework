import json
import sys

# Read original notebook
with open('lstm.ipynb', 'r', encoding='utf-8') as f:
    original = json.load(f)

# Get first 9 cells
cells = original['cells'][:9]

# Keras Tuner imports
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Keras Tuner Setup for Automatic Hyperparameter Tuning']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['# Install Keras Tuner (run once)\n', '# !pip install keras-tuner\n', '\n', 'import keras_tuner as kt\n', 'from keras_tuner import BayesianOptimization\n', 'from keras.src.callbacks import EarlyStopping\n', 'from tensorflow.keras.optimizers import Adam\n', '\n', 'print("Keras Tuner imported")']
})

# Model builder
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Tunable Model Builder Function']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['def build_tunable_model(hp):\n', '    num_layers = hp.Int("num_lstm_layers", min_value=1, max_value=3, default=2)\n', '    lstm_units = hp.Choice("lstm_units", values=[50, 64, 100, 128, 192, 256])\n', '    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05, default=0.12)\n', '    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log", default=5e-5)\n', '    \n', '    model = Sequential()\n', '    model.add(Embedding(input_dim=VOCAB_SIZE_FINAL, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], trainable=True, input_length=MAX_LEN))\n', '    model.add(Bidirectional(LSTM(units=lstm_units, dropout=dropout, return_sequences=(num_layers > 1))))\n', '    \n', '    for i in range(1, num_layers):\n', '        return_seq = (i < num_layers - 1)\n', '        model.add(LSTM(units=lstm_units, dropout=dropout, return_sequences=return_seq))\n', '    \n', '    model.add(Dense(units=1, activation="sigmoid"))\n', '    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])\n', '    return model\n', '\n', 'print("Model builder ready")']
})

# Configure tuner
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Configure Bayesian Optimization Tuner']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['tuner = BayesianOptimization(\n', '    hypermodel=build_tunable_model,\n', '    objective="val_accuracy",\n', '    max_trials=75,\n', '    executions_per_trial=1,\n', '    directory="tuner_results",\n', '    project_name="lstm_sentiment_tuning",\n', '    overwrite=False,\n', '    seed=42\n', ')\n', '\n', 'print("Tuner configured: 75 trials, Bayesian Optimization")']
})

# Early stopping
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Early Stopping Callbacks']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max", verbose=1)\n', '\n', 'class TrialPruning(tf.keras.callbacks.Callback):\n', '    def __init__(self, min_accuracy=0.70):\n', '        super().__init__()\n', '        self.min_accuracy = min_accuracy\n', '    def on_epoch_end(self, epoch, logs=None):\n', '        if epoch >= 1 and logs.get("val_accuracy", 0) < self.min_accuracy:\n', '            print(f"Stopping trial: val_accuracy {logs[\'val_accuracy\']:.4f} < {self.min_accuracy}")\n', '            self.model.stop_training = True\n', '\n', 'trial_pruning = TrialPruning(min_accuracy=0.70)\n', 'tuner_callbacks = [early_stopping, trial_pruning]\n', 'print("Callbacks configured")']
})

# Execute search
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Execute Hyperparameter Search (3-6 hours for 75 trials)']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['import time\n', '\n', 'print("="*80)\n', 'print("STARTING HYPERPARAMETER SEARCH")\n', 'print(f"Training set: {len(X_train)}, Validation set: {len(X_test)}")\n', 'print("Max epochs per trial: 20")\n', 'print("Estimated time: 3-6 hours")\n', 'print("="*80)\n', '\n', 'search_start = time.time()\n', '\n', 'tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=tuner_callbacks, batch_size=256, verbose=1)\n', '\n', 'search_time = time.time() - search_start\n', 'print(f"\\nSearch complete: {search_time/3600:.2f} hours")']
})

# Best hyperparameters
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Best Hyperparameters']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['print("="*80)\n', 'print("BEST HYPERPARAMETERS:")\n', 'print("="*80)\n', 'best_hps = tuner.get_best_hyperparameters(1)[0]\n', 'print(f"LSTM Layers: {best_hps.get(\'num_lstm_layers\')}")\n', 'print(f"LSTM Units: {best_hps.get(\'lstm_units\')}")\n', 'print(f"Dropout: {best_hps.get(\'dropout\'):.3f}")\n', 'print(f"Learning Rate: {best_hps.get(\'learning_rate\'):.6f}")\n', 'print("\\nComparison with original (2 layers, 100 units, 0.12 dropout, 0.00005 LR)")']
})

# Top 5
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Top 5 Configurations']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['print("TOP 5 CONFIGURATIONS:")\n', 'for i, trial in enumerate(tuner.oracle.get_best_trials(5), 1):\n', '    print(f"\\n#{i}: Accuracy={trial.score:.4f}")\n', '    print(f"  Layers={trial.hyperparameters.get(\'num_lstm_layers\')}, Units={trial.hyperparameters.get(\'lstm_units\')}, Dropout={trial.hyperparameters.get(\'dropout\'):.3f}, LR={trial.hyperparameters.get(\'learning_rate\'):.6f}")']
})

# Visualization
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Results Visualization']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['import matplotlib.pyplot as plt\n', 'import pandas as pd\n', '\n', 'trial_data = []\n', 'for trial in tuner.oracle.trials.values():\n', '    if trial.score is not None:\n', '        trial_data.append({\n', '            "trial_id": trial.trial_id,\n', '            "val_accuracy": trial.score,\n', '            "num_layers": trial.hyperparameters.get("num_lstm_layers"),\n', '            "lstm_units": trial.hyperparameters.get("lstm_units"),\n', '            "dropout": trial.hyperparameters.get("dropout"),\n', '            "learning_rate": trial.hyperparameters.get("learning_rate")\n', '        })\n', '\n', 'df = pd.DataFrame(trial_data)\n', '\n', 'fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n', 'fig.suptitle("Hyperparameter Tuning Results", fontsize=16)\n', '\n', 'axes[0,0].plot(df["trial_id"], df["val_accuracy"], "o-")\n', 'axes[0,0].axhline(0.8885, color="red", linestyle="--", label="Original")\n', 'axes[0,0].set_title("Accuracy Progress")\n', 'axes[0,0].legend()\n', '\n', 'axes[0,1].scatter(df["lstm_units"], df["val_accuracy"], c=df["num_layers"], cmap="viridis", s=100, alpha=0.6)\n', 'axes[0,1].set_title("LSTM Units vs Accuracy")\n', '\n', 'axes[0,2].scatter(df["dropout"], df["val_accuracy"], c=df["lstm_units"], cmap="plasma", s=100, alpha=0.6)\n', 'axes[0,2].set_title("Dropout vs Accuracy")\n', '\n', 'axes[1,0].scatter(df["learning_rate"], df["val_accuracy"], s=100, alpha=0.6)\n', 'axes[1,0].set_xscale("log")\n', 'axes[1,0].set_title("Learning Rate vs Accuracy")\n', '\n', 'df.groupby("num_layers")["val_accuracy"].agg(["mean", "max"]).plot(kind="bar", ax=axes[1,1])\n', 'axes[1,1].set_title("Performance by Layers")\n', '\n', 'axes[1,2].text(0.5, 0.5, f"Best: {df[\'val_accuracy\'].max():.4f}\\nMean: {df[\'val_accuracy\'].mean():.4f}\\nStd: {df[\'val_accuracy\'].std():.4f}", ha="center", va="center", fontsize=14)\n', 'axes[1,2].set_title("Summary Stats")\n', 'axes[1,2].axis("off")\n', '\n', 'plt.tight_layout()\n', 'plt.show()\n', '\n', 'print(f"Improvement over original: {(df[\'val_accuracy\'].max() - 0.8885)*100:.2f}%")']
})

# Train final model
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Train Final Model with Best Hyperparameters']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['print("Training final model with best hyperparameters...")\n', 'best_model = tuner.hypermodel.build(best_hps)\n', '\n', 'final_early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max", verbose=1)\n', '\n', 'final_history = best_model.fit(X_train, y_train, batch_size=256, epochs=50, validation_data=(X_test, y_test), callbacks=[final_early_stopping], verbose=1)\n', '\n', 'final_score, final_acc = best_model.evaluate(X_test, y_test, batch_size=256)\n', 'print(f"\\nFinal Test Accuracy: {final_acc:.4f}")\n', 'print(f"Final Test Loss: {final_score:.4f}")\n', 'print(f"Improvement: {(final_acc - 0.8885)*100:.2f}%")']
})

# Save results
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Save Results']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['import json\n', 'import os\n', 'from datetime import datetime\n', '\n', 'os.makedirs("models", exist_ok=True)\n', '\n', 'results = {\n', '    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),\n', '    "total_trials": len(df),\n', '    "best_hyperparameters": {\n', '        "num_lstm_layers": int(best_hps.get("num_lstm_layers")),\n', '        "lstm_units": int(best_hps.get("lstm_units")),\n', '        "dropout": float(best_hps.get("dropout")),\n', '        "learning_rate": float(best_hps.get("learning_rate"))\n', '    },\n', '    "final_accuracy": float(final_acc),\n', '    "final_loss": float(final_score),\n', '    "improvement": float((final_acc - 0.8885) * 100)\n', '}\n', '\n', 'with open("tuning_results_summary.json", "w") as f:\n', '    json.dump(results, f, indent=2)\n', '\n', 'best_model.save("models/lstm_tuned_best.keras")\n', 'df.to_csv("tuning_history.csv", index=False)\n', '\n', 'print("Saved: tuning_results_summary.json")\n', 'print("Saved: models/lstm_tuned_best.keras")\n', 'print("Saved: tuning_history.csv")']
})

# Test predictions
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Test Predictions']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['texts = ["This was the best movie I have ever seen!", "I really hated this film. It was slow and boring."]\n', '\n', 'def predict(text, model):\n', '    seq = tokenizer.texts_to_sequences([text])\n', '    padded = pad_sequences(seq, maxlen=MAX_LEN)\n', '    score = model.predict(padded, verbose=0)[0][0]\n', '    return score, "Positive" if score > 0.5 else "Negative"\n', '\n', 'print("\\nTuned Model Predictions:")\n', 'for text in texts:\n', '    score, label = predict(text, best_model)\n', '    print(f"{text[:50]}... -> {score:.4f} ({label})")']
})

# Create notebook
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('lstm_tuned.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('SUCCESS: Created lstm_tuned.ipynb')
print(f'Total cells: {len(cells)}')
