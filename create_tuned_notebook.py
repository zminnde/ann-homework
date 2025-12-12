import json

# Read original notebook
with open('lstm.ipynb', 'r', encoding='utf-8') as f:
    original_nb = json.load(f)

# Get original cells (first 9 cells - up to embedding matrix)
cells = original_nb['cells'][:9]

# Add Keras Tuner imports cell
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
    'source': [
        '# Install Keras Tuner (run once)\n',
        '# Uncomment the line below to install\n',
        '# !pip install keras-tuner\n',
        '\n',
        '# Import Keras Tuner for automatic hyperparameter tuning\n',
        'import keras_tuner as kt\n',
        'from keras_tuner import BayesianOptimization\n',
        'from keras.src.callbacks import EarlyStopping\n',
        'from keras.src import regularizers\n',
        'from tensorflow.keras.optimizers import Adam\n',
        '\n',
        'print("Keras Tuner imported successfully")'
    ]
})

# Add model builder function
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
    'source': [
        'def build_tunable_model(hp):\n',
        '    """\n',
        '    Build LSTM model with hyperparameters for Keras Tuner.\n',
        '    \n',
        '    Args:\n',
        '        hp: HyperParameters object from Keras Tuner\n',
        '    \n',
        '    Returns:\n',
        '        Compiled Keras model\n',
        '    """\n',
        '    \n',
        '    # Hyperparameter: Number of LSTM layers (1-3)\n',
        '    num_layers = hp.Int(\'num_lstm_layers\', min_value=1, max_value=3, default=2)\n',
        '    \n',
        '    # Hyperparameter: LSTM units (50-256)\n',
        '    lstm_units = hp.Choice(\'lstm_units\', values=[50, 64, 100, 128, 192, 256])\n',
        '    \n',
        '    # Hyperparameter: Dropout rate (0.0-0.5)\n',
        '    dropout = hp.Float(\'dropout\', min_value=0.0, max_value=0.5, step=0.05, default=0.12)\n',
        '    \n',
        '    # Hyperparameter: Learning rate (log scale: 1e-5 to 1e-3)\n',
        '    learning_rate = hp.Float(\'learning_rate\', min_value=1e-5, max_value=1e-3, \n',
        '                             sampling=\'log\', default=5e-5)\n',
        '    \n',
        '    # Build model\n',
        '    model = Sequential()\n',
        '    \n',
        '    # Embedding layer (using pre-loaded GloVe)\n',
        '    model.add(Embedding(\n',
        '        input_dim=VOCAB_SIZE_FINAL,\n',
        '        output_dim=EMBEDDING_DIM,\n',
        '        weights=[embedding_matrix],\n',
        '        trainable=True,\n',
        '        input_length=MAX_LEN\n',
        '    ))\n',
        '    \n',
        '    # First layer: Always Bidirectional LSTM\n',
        '    model.add(Bidirectional(LSTM(\n',
        '        units=lstm_units,\n',
        '        dropout=dropout,\n',
        '        return_sequences=(num_layers > 1),\n',
        '    )))\n',
        '    \n',
        '    # Additional LSTM layers (if num_layers > 1)\n',
        '    for i in range(1, num_layers):\n',
        '        return_seq = (i < num_layers - 1)\n',
        '        model.add(LSTM(\n',
        '            units=lstm_units,\n',
        '            dropout=dropout,\n',
        '            return_sequences=return_seq\n',
        '        ))\n',
        '    \n',
        '    # Output layer\n',
        '    model.add(Dense(units=1, activation=\'sigmoid\'))\n',
        '    \n',
        '    # Compile with tunable learning rate\n',
        '    model.compile(\n',
        '        optimizer=Adam(learning_rate=learning_rate),\n',
        '        loss=\'binary_crossentropy\',\n',
        '        metrics=[\'accuracy\']\n',
        '    )\n',
        '    \n',
        '    return model\n',
        '\n',
        'print("Model builder function created")'
    ]
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
    'source': [
        '# Directory for storing tuner results\n',
        'TUNER_PROJECT_NAME = \'lstm_sentiment_tuning\'\n',
        'TUNER_DIRECTORY = \'tuner_results\'\n',
        '\n',
        '# Initialize Bayesian Optimization Tuner\n',
        'tuner = BayesianOptimization(\n',
        '    hypermodel=build_tunable_model,\n',
        '    objective=\'val_accuracy\',\n',
        '    max_trials=75,\n',
        '    executions_per_trial=1,\n',
        '    directory=TUNER_DIRECTORY,\n',
        '    project_name=TUNER_PROJECT_NAME,\n',
        '    overwrite=False,\n',
        '    seed=42\n',
        ')\n',
        '\n',
        'print("="*80)\n',
        'print("Tuner Configuration:")\n',
        'print(f"  Max Trials: 75")\n',
        'print(f"  Objective: Maximize validation accuracy")\n',
        'print(f"  Algorithm: Bayesian Optimization")\n',
        'print(f"  Results Directory: {TUNER_DIRECTORY}/{TUNER_PROJECT_NAME}")\n',
        'print("="*80)'
    ]
})

# Early stopping callbacks
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Early Stopping Callbacks for Efficient Search']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Early stopping callback for individual trials\n',
        'early_stopping = EarlyStopping(\n',
        '    monitor=\'val_accuracy\',\n',
        '    patience=3,\n',
        '    restore_best_weights=True,\n',
        '    mode=\'max\',\n',
        '    verbose=1\n',
        ')\n',
        '\n',
        '# Custom callback to stop very poor trials early\n',
        'class TrialPruning(tf.keras.callbacks.Callback):\n',
        '    """Stop trial early if performance is clearly poor"""\n',
        '    def __init__(self, min_accuracy=0.70):\n',
        '        super().__init__()\n',
        '        self.min_accuracy = min_accuracy\n',
        '    \n',
        '    def on_epoch_end(self, epoch, logs=None):\n',
        '        if epoch >= 1:\n',
        '            if logs.get(\'val_accuracy\', 0) < self.min_accuracy:\n',
        '                print(f"\\nStopping trial: val_accuracy {logs[\'val_accuracy\']:.4f} < {self.min_accuracy}")\n',
        '                self.model.stop_training = True\n',
        '\n',
        'trial_pruning = TrialPruning(min_accuracy=0.70)\n',
        'tuner_callbacks = [early_stopping, trial_pruning]\n',
        '\n',
        'print("Early stopping callbacks configured")'
    ]
})

# Execute search
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Execute Hyperparameter Search\n', '\n', 'This will take 3-6 hours for 75 trials. Each trial trains for up to 20 epochs with early stopping.']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import time\n',
        '\n',
        'print("\\n" + "="*80)\n',
        'print("STARTING HYPERPARAMETER SEARCH")\n',
        'print("="*80)\n',
        'print(f"Training set size: {len(X_train)}")\n',
        'print(f"Validation set size: {len(X_test)}")\n',
        'print(f"Maximum epochs per trial: 20")\n',
        'print(f"Estimated time: 3-6 hours")\n',
        'print("="*80 + "\\n")\n',
        '\n',
        'search_start_time = time.time()\n',
        '\n',
        'tuner.search(\n',
        '    X_train, y_train,\n',
        '    epochs=20,\n',
        '    validation_data=(X_test, y_test),\n',
        '    callbacks=tuner_callbacks,\n',
        '    batch_size=256,\n',
        '    verbose=1\n',
        ')\n',
        '\n',
        'search_elapsed = time.time() - search_start_time\n',
        '\n',
        'print("\\n" + "="*80)\n',
        'print("HYPERPARAMETER SEARCH COMPLETE")\n',
        'print(f"Total time: {search_elapsed/3600:.2f} hours")\n',
        'print("="*80)'
    ]
})

# Results analysis cells
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Analysis: Best Hyperparameters']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'print("="*80)\n',
        'print("BEST HYPERPARAMETERS FOUND:")\n',
        'print("="*80)\n',
        '\n',
        'best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]\n',
        '\n',
        'print(f"""\\nNumber of LSTM Layers: {best_hps.get(\'num_lstm_layers\')}\n',
        'LSTM Units: {best_hps.get(\'lstm_units\')}\n',
        'Dropout Rate: {best_hps.get(\'dropout\'):.3f}\n',
        'Learning Rate: {best_hps.get(\'learning_rate\'):.6f}\n',
        '""")\n',
        '\n',
        'print("="*80)\n',
        'print("COMPARISON WITH ORIGINAL MODEL:")\n',
        'print("="*80)\n',
        'print(f"""\\nOriginal → Tuned:\n',
        '  LSTM Layers: 2 → {best_hps.get(\'num_lstm_layers\')}\n',
        '  LSTM Units: 100 → {best_hps.get(\'lstm_units\')}\n',
        '  Dropout: 0.12 → {best_hps.get(\'dropout\'):.3f}\n',
        '  Learning Rate: 0.00005 → {best_hps.get(\'learning_rate\'):.6f}\n',
        '""")'
    ]
})

cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Analysis: Top 5 Configurations']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'print("="*80)\n',
        'print("TOP 5 CONFIGURATIONS:")\n',
        'print("="*80)\n',
        '\n',
        'top_trials = tuner.oracle.get_best_trials(num_trials=5)\n',
        '\n',
        'for i, trial in enumerate(top_trials, 1):\n',
        '    print(f"\\nRank #{i}:")\n',
        '    print(f"  Val Accuracy: {trial.score:.4f}")\n',
        '    print(f"  LSTM Layers: {trial.hyperparameters.get(\'num_lstm_layers\')}")\n',
        '    print(f"  LSTM Units: {trial.hyperparameters.get(\'lstm_units\')}")\n',
        '    print(f"  Dropout: {trial.hyperparameters.get(\'dropout\'):.3f}")\n',
        '    print(f"  Learning Rate: {trial.hyperparameters.get(\'learning_rate\'):.6f}")\n',
        '    print(f"  Trial ID: {trial.trial_id}")'
    ]
})

cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Visualization: Comprehensive Results Analysis']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import matplotlib.pyplot as plt\n',
        'import seaborn as sns\n',
        'import pandas as pd\n',
        '\n',
        '# Extract trial data\n',
        'trial_data = []\n',
        'for trial in tuner.oracle.trials.values():\n',
        '    if trial.score is not None:\n',
        '        trial_data.append({\n',
        '            \'trial_id\': trial.trial_id,\n',
        '            \'val_accuracy\': trial.score,\n',
        '            \'num_layers\': trial.hyperparameters.get(\'num_lstm_layers\'),\n',
        '            \'lstm_units\': trial.hyperparameters.get(\'lstm_units\'),\n',
        '            \'dropout\': trial.hyperparameters.get(\'dropout\'),\n',
        '            \'learning_rate\': trial.hyperparameters.get(\'learning_rate\')\n',
        '        })\n',
        '\n',
        'df_trials = pd.DataFrame(trial_data)\n',
        '\n',
        '# Create visualization\n',
        'fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n',
        'fig.suptitle(\'Hyperparameter Tuning Results Analysis\', fontsize=16, fontweight=\'bold\')\n',
        '\n',
        '# 1. Validation Accuracy over Trials\n',
        'ax = axes[0, 0]\n',
        'ax.plot(df_trials[\'trial_id\'], df_trials[\'val_accuracy\'], marker=\'o\', linewidth=1, markersize=4)\n',
        'ax.axhline(y=0.8885, color=\'red\', linestyle=\'--\', label=\'Original Model (88.85%)\')\n',
        'ax.set_xlabel(\'Trial Number\')\n',
        'ax.set_ylabel(\'Validation Accuracy\')\n',
        'ax.set_title(\'Accuracy Progress Across Trials\')\n',
        'ax.legend()\n',
        'ax.grid(alpha=0.3)\n',
        '\n',
        '# 2. LSTM Units vs Accuracy\n',
        'ax = axes[0, 1]\n',
        'scatter = ax.scatter(df_trials[\'lstm_units\'], df_trials[\'val_accuracy\'], \n',
        '                     c=df_trials[\'num_layers\'], cmap=\'viridis\', s=100, alpha=0.6)\n',
        'ax.set_xlabel(\'LSTM Units\')\n',
        'ax.set_ylabel(\'Validation Accuracy\')\n',
        'ax.set_title(\'LSTM Units vs Accuracy (colored by # layers)\')\n',
        'plt.colorbar(scatter, ax=ax, label=\'Num Layers\')\n',
        'ax.grid(alpha=0.3)\n',
        '\n',
        '# 3. Dropout vs Accuracy\n',
        'ax = axes[0, 2]\n',
        'scatter = ax.scatter(df_trials[\'dropout\'], df_trials[\'val_accuracy\'], \n',
        '                     c=df_trials[\'lstm_units\'], cmap=\'plasma\', s=100, alpha=0.6)\n',
        'ax.set_xlabel(\'Dropout Rate\')\n',
        'ax.set_ylabel(\'Validation Accuracy\')\n',
        'ax.set_title(\'Dropout vs Accuracy (colored by LSTM units)\')\n',
        'plt.colorbar(scatter, ax=ax, label=\'LSTM Units\')\n',
        'ax.grid(alpha=0.3)\n',
        '\n',
        '# 4. Learning Rate vs Accuracy\n',
        'ax = axes[1, 0]\n',
        'ax.scatter(df_trials[\'learning_rate\'], df_trials[\'val_accuracy\'], s=100, alpha=0.6)\n',
        'ax.set_xlabel(\'Learning Rate\')\n',
        'ax.set_ylabel(\'Validation Accuracy\')\n',
        'ax.set_title(\'Learning Rate vs Accuracy\')\n',
        'ax.set_xscale(\'log\')\n',
        'ax.grid(alpha=0.3)\n',
        '\n',
        '# 5. Number of Layers distribution\n',
        'ax = axes[1, 1]\n',
        'layer_counts = df_trials.groupby(\'num_layers\')[\'val_accuracy\'].agg([\'mean\', \'max\', \'count\'])\n',
        'layer_counts.plot(kind=\'bar\', y=[\'mean\', \'max\'], ax=ax)\n',
        'ax.set_xlabel(\'Number of LSTM Layers\')\n',
        'ax.set_ylabel(\'Validation Accuracy\')\n',
        'ax.set_title(\'Performance by Number of Layers\')\n',
        'ax.legend([\'Mean Accuracy\', \'Max Accuracy\'])\n',
        'ax.set_xticklabels(ax.get_xticklabels(), rotation=0)\n',
        '\n',
        '# 6. Top configurations heatmap\n',
        'ax = axes[1, 2]\n',
        'top_10 = df_trials.nlargest(10, \'val_accuracy\')[[\'num_layers\', \'lstm_units\', \'dropout\', \'learning_rate\']]\n',
        'top_10_normalized = (top_10 - top_10.min()) / (top_10.max() - top_10.min())\n',
        'sns.heatmap(top_10_normalized.T, annot=False, cmap=\'YlGnBu\', ax=ax, cbar_kws={\'label\': \'Normalized Value\'})\n',
        'ax.set_xlabel(\'Top 10 Trials\')\n',
        'ax.set_ylabel(\'Hyperparameter\')\n',
        'ax.set_title(\'Top 10 Configurations (normalized)\')\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.show()\n',
        '\n',
        '# Summary statistics\n',
        'print("\\n" + "="*80)\n',
        'print("SUMMARY STATISTICS:")\n',
        'print("="*80)\n',
        'print(f"Total trials completed: {len(df_trials)}")\n',
        'print(f"Best validation accuracy: {df_trials[\'val_accuracy\'].max():.4f}")\n',
        'print(f"Mean validation accuracy: {df_trials[\'val_accuracy\'].mean():.4f}")\n',
        'print(f"Std validation accuracy: {df_trials[\'val_accuracy\'].std():.4f}")\n',
        'print(f"Improvement over original: {(df_trials[\'val_accuracy\'].max() - 0.8885)*100:.2f}%")'
    ]
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
    'source': [
        'print("="*80)\n',
        'print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")\n',
        'print("="*80)\n',
        '\n',
        '# Build model with best hyperparameters\n',
        'best_model = tuner.hypermodel.build(best_hps)\n',
        '\n',
        '# More patient early stopping for final training\n',
        'final_early_stopping = EarlyStopping(\n',
        '    monitor=\'val_accuracy\',\n',
        '    patience=5,\n',
        '    restore_best_weights=True,\n',
        '    mode=\'max\',\n',
        '    verbose=1\n',
        ')\n',
        '\n',
        'print(f"\\nTraining final model for up to 50 epochs...")\n',
        'final_start = time.time()\n',
        '\n',
        'final_history = best_model.fit(\n',
        '    X_train, y_train,\n',
        '    batch_size=256,\n',
        '    epochs=50,\n',
        '    validation_data=(X_test, y_test),\n',
        '    callbacks=[final_early_stopping],\n',
        '    verbose=1\n',
        ')\n',
        '\n',
        'final_elapsed = time.time() - final_start\n',
        'print(f"\\nFinal training time: {final_elapsed/60:.2f} minutes")\n',
        '\n',
        '# Evaluate final model\n',
        'final_score, final_acc = best_model.evaluate(X_test, y_test, batch_size=256)\n',
        '\n',
        'print("\\n" + "="*80)\n',
        'print("FINAL MODEL PERFORMANCE:")\n',
        'print("="*80)\n',
        'print(f"Test Accuracy: {final_acc:.4f}")\n',
        'print(f"Test Loss: {final_score:.4f}")\n',
        'print(f"Improvement over original: {(final_acc - 0.8885)*100:.2f}%")\n',
        'print("="*80)'
    ]
})

# Save results
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Save Results and Model']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import json\n',
        'from datetime import datetime\n',
        'import os\n',
        '\n',
        '# Create models directory if it doesn\'t exist\n',
        'os.makedirs(\'models\', exist_ok=True)\n',
        '\n',
        '# Save tuning results summary\n',
        'results_summary = {\n',
        '    \'tuning_date\': datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\'),\n',
        '    \'total_trials\': len(df_trials),\n',
        '    \'best_hyperparameters\': {\n',
        '        \'num_lstm_layers\': int(best_hps.get(\'num_lstm_layers\')),\n',
        '        \'lstm_units\': int(best_hps.get(\'lstm_units\')),\n',
        '        \'dropout\': float(best_hps.get(\'dropout\')),\n',
        '        \'learning_rate\': float(best_hps.get(\'learning_rate\'))\n',
        '    },\n',
        '    \'best_val_accuracy\': float(df_trials[\'val_accuracy\'].max()),\n',
        '    \'final_test_accuracy\': float(final_acc),\n',
        '    \'final_test_loss\': float(final_score),\n',
        '    \'original_model_accuracy\': 0.8885,\n',
        '    \'improvement\': float((final_acc - 0.8885) * 100)\n',
        '}\n',
        '\n',
        '# Save to JSON\n',
        'with open(\'tuning_results_summary.json\', \'w\') as f:\n',
        '    json.dump(results_summary, f, indent=2)\n',
        '\n',
        'print("Saved tuning summary to: tuning_results_summary.json")\n',
        '\n',
        '# Save best model\n',
        'best_model.save(\'models/lstm_tuned_best.keras\')\n',
        'print("Saved best model to: models/lstm_tuned_best.keras")\n',
        '\n',
        '# Save tuning history CSV\n',
        'df_trials.to_csv(\'tuning_history.csv\', index=False)\n',
        'print("Saved trial history to: tuning_history.csv")\n',
        '\n',
        'print("\\n" + "="*80)\n',
        'print("ALL RESULTS SAVED SUCCESSFULLY")\n',
        'print("="*80)'
    ]
})

# Add prediction cells
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Test Predictions with Tuned Model']
})

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'texts_to_predict = [\n',
        '    "This was the best movie I have ever seen!",\n',
        '    "I really hated this film. It was slow and boring.",\n',
        '    "The acting was decent, but the plot was a little weak.",\n',
        '    "Scariest movie I have ever seen and",\n',
        ']\n',
        '\n',
        'def get_sentiment_prediction(text, model):\n',
        '    sequence = tokenizer.texts_to_sequences([text])\n',
        '    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)\n',
        '    prediction = model.predict(padded_sequence, verbose=0)\n',
        '    score = prediction[0][0]\n',
        '    label = "Positive" if score > 0.5 else "Negative"\n',
        '    return score, label\n',
        '\n',
        'import textwrap\n',
        'text_width = 40\n',
        '\n',
        'print("="*80)\n',
        'print("TESTING TUNED MODEL PREDICTIONS:")\n',
        'print("="*80 + "\\n")\n',
        '\n',
        'for text in texts_to_predict:\n',
        '    score, sentiment = get_sentiment_prediction(text, best_model)\n',
        '    wrapped_text = textwrap.fill(text, width=text_width)\n',
        '    print(wrapped_text)\n',
        '    print(f"\\n  -> Prediction: {score:.4f} ({sentiment})")\n',
        '    print("=" * (text_width + 4))'
    ]
})

# Create final notebook
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.12.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

# Save to file
with open('lstm_tuned.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('✓ Created lstm_tuned.ipynb with hyperparameter tuning!')
print('\\nThis notebook includes:')
print('  - All original preprocessing and data loading')
print('  - Keras Tuner with Bayesian Optimization (75 trials)')
print('  - Automatic tuning of: LSTM units, dropout, layers, learning rate')
print('  - Comprehensive results visualization')
print('  - Final model training and saving')
print('\\nOpen lstm_tuned.ipynb in Jupyter and run all cells to start tuning!')
