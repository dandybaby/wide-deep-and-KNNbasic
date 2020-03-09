from surprise import Dataset
from surprise import KNNBasic, Reader
from surprise.model_selection import cross_validate
#数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)
train_set = data.build_full_trainset()

#训练模型
algo = KNNBasic()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)