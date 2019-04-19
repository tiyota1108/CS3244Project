import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
from keras.models import model_from_json
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load the json file that contains the model's structure
with open('model_structure2.json', 'r') as f:
    model_structure = f.read()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights2.h5")
test_images = joblib.load("x_test2.dat")
test_labels = joblib.load("y_test2.dat")
test_labels1d=np.zeros((1,len(test_labels)))
for i,list in enumerate(test_labels):
    test_labels1d[0,i]=list.tolist().index(1)


# print("")
predictions = model.predict(test_images)
predictions1d = np.argmax(predictions, axis=1)
test_labels1d=test_labels1d[0]

MODEL_NAME = 'transfer learning vgg16'
CATEGORIES = ['ReplenishmentVessel',
'SupplyVessel',
'HeavyLoadCarrier',
'Tug',
'FireFightingVessel',
'Platform',
'TrainingVessel',
'Container',
'Reefer',
'Passenger']

#Confusion matrix
cm = confusion_matrix(test_labels1d, predictions1d)
plt.figure(figsize = (10,10))
heatmap = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, annot_kws={'fontsize':14})
heatmap.yaxis.set_ticklabels(CATEGORIES, rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(CATEGORIES, rotation=45, ha='right')
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
plt.title('Confusion matrix for {}'.format(MODEL_NAME), fontsize = 16)
plt.savefig('{} confusion matrix'.format(MODEL_NAME), bbox_inches = 'tight')

#Generate and save report
report = classification_report(test_labels1d, predictions1d, target_names=CATEGORIES)
print(report)
f = open('{} classification report.txt'.format(MODEL_NAME), 'w')
f.write(report)
f.close()
