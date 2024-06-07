import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pokemon_stats = pd.read_csv('pokemon_data2.csv')  
battle_results = pd.read_csv('combats.csv')
print(pokemon_stats.info())
print(battle_results.info())
#數據清理及處理
pokemon_stats['Type 2'].fillna('None', inplace=True)##
#把類別轉成數字
le = preprocessing.LabelEncoder()
pokemon_stats['Type 1'] = le.fit_transform(pokemon_stats['Type 1'])
pokemon_stats['Type 2'] = le.fit_transform(pokemon_stats['Type 1'])
pokemon_stats['Legendary'] = pokemon_stats['Legendary'].astype(int)
# 合併資料
merged_data = pd.merge(battle_results, pokemon_stats, left_on='p1_id', right_on='id', how='left')
merged_data = pd.merge(merged_data, pokemon_stats, left_on='p2_id', right_on='id', suffixes=('_p1', '_p2'), how='left')

#刪除重複的id
merged_data.drop(columns=['id_p1','id_p2'])

# 創建屬性差值特徵
merged_data['HP_diff'] = merged_data['HP_p1'] - merged_data['HP_p2']
merged_data['Attack_diff'] = merged_data['Attack_p1'] - merged_data['Attack_p2']
merged_data['Defense_diff'] = merged_data['Defense_p1'] - merged_data['Defense_p2']
merged_data['Sp_Atk_diff'] = merged_data['Sp. Atk_p1'] - merged_data['Sp. Atk_p2']
merged_data['Sp_Def_diff'] = merged_data['Sp. Def_p1'] - merged_data['Sp. Def_p2']
merged_data['Speed_diff'] = merged_data['Speed_p1'] - merged_data['Speed_p2']

# 創建屬性比值特徵 加上1e-5是為了避免除數為0
merged_data['HP_ratio'] = merged_data['HP_p1'] / (merged_data['HP_p2'] + 1e-5)
merged_data['Attack_ratio'] = merged_data['Attack_p1'] / (merged_data['Attack_p2'] + 1e-5)
merged_data['Defense_ratio'] = merged_data['Defense_p1'] / (merged_data['Defense_p2'] + 1e-5)
merged_data['Sp_Atk_ratio'] = merged_data['Sp. Atk_p1'] / (merged_data['Sp. Atk_p2'] + 1e-5)
merged_data['Sp_Def_ratio'] = merged_data['Sp. Def_p1'] / (merged_data['Sp. Def_p2'] + 1e-5)
merged_data['Speed_ratio'] = merged_data['Speed_p1'] / (merged_data['Speed_p2'] + 1e-5)

# 是否是傳說寶可夢
merged_data['Legendary_diff'] = merged_data['Legendary_p1'] - merged_data['Legendary_p2']

features = ['HP_diff', 'Attack_diff', 'Defense_diff', 'Sp_Atk_diff', 'Sp_Def_diff', 'Speed_diff', 
            'HP_ratio', 'Attack_ratio', 'Defense_ratio', 'Sp_Atk_ratio', 'Sp_Def_ratio', 'Speed_ratio', 
            'Legendary_diff', 'Type 1_p1', 'Type 2_p1', 'Type 1_p2', 'Type 2_p2']

X = merged_data[features]
y = (merged_data['winner_id'] == merged_data['p1_id']) # 使用(winner_id是否等於p1_id)作為目標變數

# 將資料分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 建立模型
model = RandomForestClassifier(n_estimators=50)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 評估模型
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print('Train Accuracy:',accuracy_train)
print('Test Accuracy:', accuracy_test)