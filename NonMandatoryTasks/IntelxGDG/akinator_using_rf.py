# I couldnt find a better dataset for this problem, its a very small dataset sorry guys !!
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


import os
import pandas as pd

#This part is chatgpt cuz i dont know how to work with files!!
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "characters.csv")

df = pd.read_csv(file_path)
print(df.head())

df = df.fillna(0)


target_col = df.columns[-1]
le = LabelEncoder()
df['character_label'] = le.fit_transform(df[target_col])


feature_cols = [col for col in df.columns if col not in ['name', 'character_label']]
X = df[feature_cols].astype('float32').values

y = df['character_label'].values.astype('int32')


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


rf = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)


top1_acc = accuracy_score(y_test, y_pred)
print(f"Top-1 Accuracy: {top1_acc:.4f}")


y_proba = rf.predict_proba(X_test)
top3_acc = np.mean([y_test[i] in np.argsort(y_proba[i])[-3:] for i in range(len(y_test))])
print(f"Top-3 Accuracy: {top3_acc:.4f}")



