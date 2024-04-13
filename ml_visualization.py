import matplotlib.pyplot as plt
import numpy as np
model_scores = {'wind_speed': {'Naive Bayes': {'Training': 0.8168672360248447, 'Validation': 0.8250363901018923}, 'KNN': {'Training': 0.8117236024844721, 'Validation': 0.7353711790393013}, 'Random Forest': {'Training': 0.9333268633540373, 'Validation': 0.9248908296943231}, 'Neural Network': {'Training': 0.8307453416149069, 'Validation': 0.8189228529839884}}, 'temperature': {'Naive Bayes': {'Training': 0.8144409937888198, 'Validation': 0.8151382823871907}, 'KNN': {'Training': 0.8118206521739131, 'Validation': 0.7141193595342067}, 'Random Forest': {'Training': 0.9126552795031055, 'Validation': 0.9211062590975255}, 'Neural Network': {'Training': 0.7932841614906833, 'Validation': 0.8002911208151383}}, 'relative_humidity': {'Naive Bayes': {'Training': 0.828027950310559, 'Validation': 0.8346433770014556}, 'KNN': {'Training': 0.8208462732919255, 'Validation': 0.7481804949053857}, 'Random Forest': {'Training': 0.9257569875776398, 'Validation': 0.9324599708879184}, 'Neural Network': {'Training': 0.6527562111801242, 'Validation': 0.6614264919941776}}, 'fire_spread_rate': {'Naive Bayes': {'Training': 0.8453998447204969, 'Validation': 0.8526928675400292}, 'KNN': {'Training': 0.7861995341614907, 'Validation': 0.6882096069868996}, 'Random Forest': {'Training': 0.8977096273291926, 'Validation': 0.9007278020378457}, 'Neural Network': {'Training': 0.6852678571428571, 'Validation': 0.6812227074235808}}, 'current_size': {'Naive Bayes': {'Training': 0.6418866459627329, 'Validation': 0.6398835516739447}, 'KNN': {'Training': 0.7168090062111802, 'Validation': 0.5909752547307132}, 'Random Forest': {'Training': 0.6919642857142857, 'Validation': 0.6850072780203784}, 'Neural Network': {'Training': 0.5850155279503105, 'Validation': 0.586608442503639}}
    #this is copy pasted from fire_model2.py
}

features_removed = list(model_scores.keys())
models = list(model_scores[features_removed[0]].keys())
metrics = list(model_scores[features_removed[0]][models[0]].keys())

fig, axes = plt.subplots(len(models), len(metrics), figsize=(15, 10), sharey='row')

for i, model in enumerate(models):
    for j, metric in enumerate(metrics):
        scores = [model_scores[feature][model][metric] for feature in features_removed]
        ax = axes[i, j]
        ax.bar(features_removed, scores, color='skyblue', edgecolor='black')
        ax.set_title(f'{model} - {metric}')
        ax.set_ylim(0.5, 1.0) 
        
        ax.set_yticks(np.arange(0.5, 1.01, 0.25))
        ax.set_yticklabels(['0.5', '0.75', '1.0'])

        ax.set_yticks(np.arange(0.5, 1.01, 0.05), minor=True)
        
        ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5)
        
        ax.tick_params(axis='x', rotation=45)

for ax in axes[:,0]:
    ax.set_ylabel('Score')

for ax in axes[-1,:]:
    ax.set_xlabel('Feature Removed')

plt.suptitle('Model Performance with Features Removed')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()