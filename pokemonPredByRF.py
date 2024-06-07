import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
# 讀取資料
pokemon_stats = pd.read_csv('pokemon_data2.csv')  
battle_results = pd.read_csv('combats.csv')

#把類別轉成數字
le = preprocessing.LabelEncoder()
pokemon_stats['Type 1'] = le.fit_transform(pokemon_stats['Type 1'])
pokemon_stats['Type 2'] = le.fit_transform(pokemon_stats['Type 1'])

# 合併資料
merged_data = pd.merge(battle_results, pokemon_stats, left_on='p1_id', right_on='id', how='left')
merged_data = pd.merge(merged_data, pokemon_stats, left_on='p2_id', right_on='id', suffixes=('_p1', '_p2'), how='left')

# 建立特徵 Type 1	Type 2	HP	Attack	Defense	Sp. Atk	Sp. Def	Speed
features = ['Type 1_p1','Type 2_p1', 'HP_p1', 'Attack_p1', 'Defense_p1', 'Sp. Atk_p1', 'Sp. Def_p1', 'Speed_p1',
'Type 1_p2','Type 2_p2', 'HP_p2', 'Attack_p2', 'Defense_p2', 'Sp. Atk_p2', 'Sp. Def_p2', 'Speed_p2',]

X = merged_data[features]
y = (merged_data['winner_id'] == merged_data['p1_id']) # 使用winner_id是否等於p1_id作為目標變數

# 將資料分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
#print('Confusion Matrix:')
#print(conf_matrix)
