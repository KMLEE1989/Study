import warnings
warnings.filterwarnings('ignore')

import pandas as pd

path = '../Study/Hackarthon/dacon/open/'

train = pd.read_csv(path + 'train_data.csv')
test  = pd.read_csv(path + "test_data.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

print(train.sample(3))

train = train.dropna(how='any') # Null 값이 존재하는 행 제거
train = train.reset_index(drop=True)
print(train.isnull().values.any()) # Null 값이 존재하는지 확인

test = test.dropna(how='any') # Null 값이 존재하는 행 제거
test = test.reset_index(drop=True)
print(test.isnull().values.any()) # Null 값이 존재하는지 확인

print("premise 최대 길이:", train['premise'].map(len).max())
print("hypothesis 최대 길이:", train['hypothesis'].map(len).max())

print("premise 최대 길이:", test['premise'].map(len).max())
print("hypothesis 최대 길이:", test['hypothesis'].map(len).max())

max_seq_len = 100

valid = train[20000:]
train = train[:20000]

from transformers import AutoTokenizer

model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from tqdm import tqdm
import numpy as np

def convert_examples_to_features(sent_list1, sent_list2, max_seq_len, tokenizer):
    
    input_ids, attention_masks, token_type_ids = [], [], []

    for sent1, sent2 in tqdm(zip(sent_list1, sent_list2), total=len(sent_list1)):
        encoding_result = tokenizer.encode_plus(sent1, sent2, 
                                                max_length=max_seq_len, 
                                                pad_to_max_length=True)

        input_ids.append(encoding_result['input_ids'])
        attention_masks.append(encoding_result['attention_mask'])
        token_type_ids.append(encoding_result['token_type_ids'])

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)

X_train = convert_examples_to_features(train['premise'], train['hypothesis'], 
                                       max_seq_len=max_seq_len, 
                                       tokenizer=tokenizer)

X_valid = convert_examples_to_features(valid['premise'], valid['hypothesis'], 
                                       max_seq_len=max_seq_len, 
                                       tokenizer=tokenizer)

X_test = convert_examples_to_features(test['premise'], test['hypothesis'], 
                                       max_seq_len=max_seq_len, 
                                       tokenizer=tokenizer)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(train['label'])
y_valid = le.transform(valid['label'])

label_idx = dict(zip(list(le.classes_), le.transform(list(le.classes_))))
label_idx

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal
from transformers import TFAutoModel

class TFBertForSequenceClassification(Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFAutoModel.from_pretrained(model_name, 
                                                num_labels=3, 
                                                from_pt=True)
        self.classifier = Dense(3,
                                kernel_initializer=TruncatedNormal(0.02),
                                activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids=inputs
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction
    
import tensorflow as tf

model = TFBertForSequenceClassification(model_name)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_accuracy", 
    min_delta=0.001,
    patience=3)

model.fit(
    X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid),
    callbacks = [early_stopping]
)

pred = model.predict(X_test)

pred.shape

result = [np.argmax(val) for val in pred]

out = [list(label_idx.keys())[_] for _ in result]
out[:3]

submit_file["label"] = out

submit_file.to_csv(path + "Mercury2002.csv", index=False)