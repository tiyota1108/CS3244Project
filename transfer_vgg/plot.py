import matplotlib.pyplot as plt
import json

with open('history2.json', 'r') as f:
    history_string = f.read()

history = json.loads(history_string)

# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.title('Model Accuracy by Epoch')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# # plt.show()
# plt.savefig('{} Model Accuracy by Epoch'.format('acc_graph'), bbox_inches = 'tight')

# "Loss"
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss by Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig('{} Model Loss by Epoch'.format('loss_graph'), bbox_inches = 'tight')
