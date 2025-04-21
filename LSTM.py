import pandas as pd
from time import time

sample_submission = pd.read_csv("sample_submission.csv")
calendar = pd.read_csv("calendar.csv")
sell_prices = pd.read_csv("sell_prices.csv")
sales_train_validation = pd.read_csv("sales_train_validation.csv")
sales_train_evaluation = pd.read_csv("sales_train_evaluation.csv")

# 壓縮記憶體大小
import numpy as np

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2  # 將記憶體轉為MB
    for col in df.columns:  # 讀取 dataframe 的欄位
        col_type = df[col].dtypes  # 判斷讀取出來的欄位的資料型態
        if col_type in numerics:  # 如果資料型態存在上面 numerics 型態
            c_min = df[col].min() # 搜尋全部中的最大與最小值
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':  # 擷取這個資料型態的前三字，判斷是否為 int
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:  # 判斷c_min是否介於 int8(-128~127) 之間，若是將所有資料轉為 np.int8
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem)) # 印出記憶體壓縮後的大小與節省百分比
    return df


sales_train_validation = reduce_mem_usage(sales_train_validation)
sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)
sell_prices = reduce_mem_usage(sell_prices)
calendar = reduce_mem_usage(calendar)


# 將不會使用到的 column drop 掉
calendar = calendar.drop(["date", "weekday"], axis=1)

# 將有發生 event 的日子記下來
calendar["event"] = calendar.apply(lambda row: 1 if type(row["event_name_1"])==str else 0, axis=1)
calendar["event"] = calendar.apply(lambda row: 2 if type(row["event_name_2"])==str else row["event"], axis=1) # 保留 envent_name_1 的資訊
calendar = calendar.drop(["event_name_1", "event_type_1", "event_name_2", "event_type_2"], axis=1)

# 設計銷量
Feature_1 = sales_train_validation.drop(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1)
Feature_1 = Feature_1.to_numpy().T


# Drop 掉 部分 column
Feature_2_8 = calendar[0:1913].drop(["event", "d", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"], axis=1)
# 針對類別型態的欄位進行 one-hot encoder
Feature_2_8 = pd.get_dummies(Feature_2_8.astype(str))
# 為 Feature_2_8 這個 dataframe 新增目標 column
Feature_2_8["event"] = calendar[0:1913]["event"]
Feature_2_8["day"] = calendar[0:1913]["d"]
Feature_2_8["snap_CA"] = calendar[0:1913]["snap_CA"]
Feature_2_8["snap_TX"] = calendar[0:1913]["snap_TX"]
Feature_2_8["snap_WI"] = calendar[0:1913]["snap_WI"]
Feature_2_8 = Feature_2_8.set_index("day")  # 將 day 設為 index
Feature_2_8.head()

# 建立 day_rule 這個 mapping 規則， 建立從 day 到 wm_yr_wk 的轉換關係，
calendar_B = pd.DataFrame()
calendar_B["wm_yr_wk"] = calendar["wm_yr_wk"]
calendar_B["d"] = calendar["d"]
day_rule = calendar_B.set_index("d").to_dict()
day_rule = day_rule["wm_yr_wk"]


# 使用 melt 將 day_1, day_2, day_3, ..., day_1913 轉成 row
sales_train_validation = pd.melt(sales_train_validation, id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name="day", value_name="count")
# 透過lambda 將原本沒有 wm_yr_wk 的 sales_train_validation 生成 wm_yr_wk
sales_train_validation["wm_yr_wk"] = sales_train_validation.apply(lambda row: day_rule[row["day"]], axis=1)
# 把 sales_train_validation 和 sell_prices 合併起來
sales_train_validation = sales_train_validation.merge(sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
# 填補缺失值
sales_train_validation['sell_price'] = sales_train_validation['sell_price'].fillna(sales_train_validation.groupby("id")["sell_price"].transform('mean'))
Feature_9 = sales_train_validation.drop(["id", "day", "item_id", "dept_id", "cat_id", "store_id", "state_id", "wm_yr_wk", "count"], axis=1)
Feature_9 = Feature_9.to_numpy().reshape(-1, 30490)

# 將不會再使用到的 dataframe 所占用的記憶體清除
del sales_train_validation
del calendar
del calendar_B
del sell_prices

Features_Train = np.concatenate([Feature_1, Feature_2_8, Feature_9], axis=1)
# print(Features_Train.shape)

from sklearn.preprocessing import StandardScaler
Feature_Scaler = StandardScaler().fit(Features_Train)
Features_Train = Feature_Scaler.transform(Features_Train)


from tqdm import tqdm
# 每 ? 個 time_stamp 的資料為一組(ex: ?=28 則 day_1~day_28)，目標是預測下一個 time_stamp (ex: day_29)的所有產品銷量
time_stamps = 7
def create_dataset(data, time_stamps):
    x = []
    y = []
    for i in tqdm(range(len(data) - time_stamps)):
        x.append(data[i:i + time_stamps])
        y.append(data[i + time_stamps])
    return np.array(x), np.array(y)
x_train, y_train = create_dataset(Features_Train, time_stamps)
print(x_train.shape, y_train.shape)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Softmax
from tensorflow.keras.regularizers import l2

drop_proba = 0.5
kernel_reg = 1e-5
recurr_reg = 1e-5

LSTM_model = tf.keras.models.Sequential([
    LSTM(
        units = 128,
        activation = "tanh",
        kernel_regularizer = l2(kernel_reg), 
        recurrent_regularizer = l2(recurr_reg),
        dropout = drop_proba,
        return_sequences=True
        ),
    Dropout(drop_proba),
    LSTM(
        units = 512,
        activation = "tanh", 
        kernel_regularizer = l2(kernel_reg), 
        recurrent_regularizer = l2(recurr_reg),
        dropout = drop_proba,
        return_sequences=False
        ),
    Dense(1024, activation = "relu"),
    Dropout(drop_proba),
    Dense(y_train.shape[1])
])
LSTM_model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
LSTM_model.summary()

batch_size = 128
epochs = 40
lr = 1e-3

optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
loss = tf.keras.losses.MeanSquaredError()

LSTM_model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

def learning_rate_scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1).numpy()

callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

History = LSTM_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[callback]
)
import matplotlib.pyplot as plt

Train_Loss = History.history["loss"]
Val_Loss = History.history["val_loss"]

plt.figure(figsize=(10,6))
plt.plot(Train_Loss, label="Train Loss")
plt.plot(Val_Loss, label="Val Loss")
plt.title("Loss", fontsize=20)
plt.ylabel("Loss", fontsize=16)
plt.xlabel("Epoch", fontsize=16)
plt.show()


mae = History.history["mae"]
Val_mae = History.history["mae"]

plt.figure(figsize=(10,6))
plt.plot(mae, label="mae")
plt.plot(Val_mae, label="Val_mae")
plt.title("accuracy", fontsize=20)
plt.ylabel("accuracy", fontsize=16)
plt.xlabel("Epoch", fontsize=16)
plt.show()
