""" 학습 코드
"""

import os, sys
import random
import argparse
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

import warnings
warnings.filterwarnings('ignore')

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
# from modules.earlystoppers import LossEarlyStopper
# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)


# DATA
TRAIN_DATASET = '/DATA/train/train.csv'
TRAIN_DATA_DIR = os.path.join(PROJECT_DIR, TRAIN_DATASET)
TEST_DATASET = '/DATA/test/test.csv'
TEST_DATA_DIR = os.path.join(PROJECT_DIR, TEST_DATASET)
MODEL_NAME = 'cgb_ver_1'

# Hyperparameters
SEED = 42
FINAL_N = 540
MAX_DURATION = 5500


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	KST = timezone(timedelta(hours=9))
	#parser.add_argument('--train_serial', type=str, default = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S"))
	parser.add_argument('--EDA', action='store_true')
	parser.add_argument('--Train', action='store_true')
	parser.add_argument('--Predict', action='store_true')
	#parser.add_argument('--train_serial', type=str, default = datetime.now().strftime("%Y%m%d_%H%M%S"))
	args=parser.parse_args()


	seed_everything(SEED)




	# Set train result directory
	train_result_dir = os.path.join(PROJECT_DIR, 'results', 'model')
	os.makedirs(train_result_dir, exist_ok=True)

	# Load data
	df = pd.read_csv(TRAIN_DATA_DIR)

	#Replace N or Y


	test_df = pd.read_csv(TEST_DATA_DIR)

	if args.EDA is True:
		print("DataFrame Head")
		print(df.head())

		print("NaN Check")
		print(df.isnull().sum())

		dur = df['duration']
		dead_or_alive = df['dead']


		print("duration chk")
		print(dur)



		print("Dead count")
		print(dead_or_alive.value_counts())
		
		print("DataFrame shape")
		print(df.shape)
		
		print("DataFrame Test head")
		print(test_df.head())

		print("NaN test chk")
		print(test_df.isnull().sum())




	#scaler
	scaler = MinMaxScaler()


	#preprocess
	weight_mean = df['weight'].mean()
	height_mean = df['height'].mean()
	totaldose_mean = df['totaldose'].mean()
	radiationcnt_mean = df['radiationcnt'].mean()
	radiationperdose_mean = df['radiationperdose'].mean()

	df['weight'].fillna(weight_mean, inplace = True)
	df['height'].fillna(height_mean, inplace = True)
	df['familyhistory'].fillna("Unknown" ,inplace = True)
	df['bp'].fillna("Unknown",inplace = True) 
	df['bs'].fillna("Unknown",inplace = True)
	df['sm'].fillna("Unknown",inplace = True)
	df['cancerimagingN'].fillna(0,inplace = True)
	df.drop('deathsign',inplace = True, axis=1)
	df['surgicalcancerT'].fillna("Unknown",inplace = True)
	df['surgicalcancerN'].fillna("Unknown",inplace = True)
	df['surgicalcancerM'].fillna("Unknown",inplace = True)
	df['boundarysurgical'].fillna("Unknown",inplace = True)
	df['involvementrenal'].fillna("Unknown", inplace = True)
	df['lymphrenal'].fillna("Unknown", inplace = True)
	df['surgicalmethod'].fillna("Unknown", inplace = True)
	df['treatmethod'].fillna("Unknown", inplace = True)
	df['treatech'].fillna("Unknown", inplace = True)
	df['egfr'].fillna("Unknown", inplace = True)
	df['ros1'].fillna("Unknown", inplace = True)
	df['alk'].fillna("Unknown", inplace = True)
	df['totaldose'].fillna(totaldose_mean, inplace = True)
	df['radiationcnt'].fillna(radiationcnt_mean, inplace = True)
	df['radiationperdose'].fillna(radiationperdose_mean, inplace = True)

	#df.drop('surgicalcancerT',inplace = True, axis=1)
	#df.drop('surgicalcancerN',inplace = True, axis=1)
	#df.drop('surgicalcancerM',inplace = True, axis=1)
	#df.drop('boundarysurgical',inplace = True, axis=1)
	#df.drop('involvementrenal',inplace = True, axis=1)
	#df.drop('lymphrenal',inplace = True, axis=1)
	#df.drop('surgicalmethod',inplace = True, axis=1)

	df['BMI'] = df['weight'] / (df['height']*df['height'])
	print(df['BMI'])

	df.drop('weight',inplace=True,axis=1)
	df.drop('height',inplace=True,axis=1)
	df.drop('totaldose',inplace = True, axis=1)
	df.drop('radiationcnt',inplace = True, axis=1)
	df.drop('radiationperdose',inplace = True, axis=1)
	#df.drop('treatech',inplace = True, axis=1)
	#df.drop('ros1',inplace = True, axis=1)


	onehot_cols =['familyhistory','bp','bs','sm','locationcancer','classificationcancer','surgicalcancerT','surgicalcancerN',
	'surgicalcancerM','boundarysurgical','involvementrenal','lymphrenal','surgicalmethod','treatmethod','treatech','egfr','ros1','alk']


	#df = pd.get_dummies(df, columns=onehot_cols)
	
	print(df.isnull().sum())
	print(df.shape)
	print(df.columns)

	df.replace({'N': 0, 'Y': 1, 'Unknown' : 2, 'M' : 0, 'F' : 1}, inplace = True)


	col = list(df.columns)
	col.remove('duration')
	col.remove('dead')
	df_train, df_val = train_test_split(df, test_size = 0.05 ,stratify = df['dead'],random_state=SEED)
	get_target = lambda df: (df['duration'].values, df['dead'].values)

	df_train = df_train.astype('float32')
	df_val = df_val.astype('float32')

	train_y = df_train.apply(lambda x: (x.dead, x.duration), axis=1).to_numpy(dtype=[('dead', 'bool'), ('duration', 'float64')])
	val_y = df_val.apply(lambda x: (x.dead, x.duration), axis=1).to_numpy(dtype=[('dead', 'bool'), ('duration', 'float64')])

	chk_val_y =get_target(df_val)

	#train_y = get_target(df_train)
	#val_y = get_target(df_val)
	#train_y = train_y.astype(np.float32)
	#val_y = val_y.astype(np.float32)
	df_train.drop(['duration','dead'],axis=1,inplace=True)
	df_val.drop(['duration','dead'],axis=1,inplace=True)
	train_x =scaler.fit_transform(df_train)
	print(train_x.shape)
	val_x = scaler.transform(df_val)

	if args.Train is True:
		best_c = 0
		best_n = 0
		for i in range(100):
			n = 10+i*10
			cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=n, random_state=SEED)
			cgb.fit(train_x,train_y)
			c_index = cgb.score(val_x,val_y)
			print("C-index : " + str(c_index))
			if best_c < c_index:
				best_c = c_index
				best_n = n
				print("update n : " + str(n))
		print("Best c-index" + str(best_c))
		print("Best n_estimators "+str(best_n))
		cgb_best = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=best_n, random_state=SEED)
		cgb_best.fit(train_x,train_y)
		print(cgb.score(val_x,val_y))
		surv = cgb.predict_survival_function(val_x, return_array=True)
		surv_df = pd.DataFrame(surv)
		surv_df = surv_df.transpose()
		ss = pd.DataFrame(index=list(range(int(max(surv_df.index) + 1))))
		ss = ss.join(surv_df)
		ss.fillna(method = 'bfill',inplace = True)
		# y_val[0]: duration , y_val[1]: event
		ev = EvalSurv(ss, chk_val_y[0], chk_val_y[1], censor_surv='km')

		print("c-index : " , ev.concordance_td(method='adj_antolini'))
		

	if args.Predict is True:

		test_df = pd.read_csv(TEST_DATA_DIR)
		test_df['weight'].fillna(weight_mean, inplace = True)
		test_df['height'].fillna(height_mean, inplace = True)
		test_df['familyhistory'].fillna("Unknown" ,inplace = True)
		test_df['bp'].fillna("Unknown",inplace = True) 
		test_df['bs'].fillna("Unknown",inplace = True)
		test_df['sm'].fillna("Unknown",inplace = True)
		test_df['cancerimagingN'].fillna("Unknown",inplace = True)
		test_df['surgicalcancerT'].fillna("Unknown",inplace = True)
		test_df['surgicalcancerN'].fillna("Unknown",inplace = True)
		test_df['surgicalcancerM'].fillna("Unknown",inplace = True)
		test_df['boundarysurgical'].fillna("Unknown",inplace = True)
		test_df['involvementrenal'].fillna("Unknown", inplace = True)
		test_df['lymphrenal'].fillna("Unknown", inplace = True)
		test_df['surgicalmethod'].fillna("Unknown", inplace = True)
		test_df['treatmethod'].fillna("Unknown", inplace = True)
		test_df['treatech'].fillna("Unknown", inplace = True)
		test_df['egfr'].fillna("Unknown", inplace = True)
		test_df['ros1'].fillna("Unknown", inplace = True)
		test_df['alk'].fillna("Unknown", inplace = True)
		test_df['totaldose'].fillna(0, inplace = True)
		test_df['radiationcnt'].fillna(0, inplace = True)
		test_df['radiationperdose'].fillna(0, inplace = True)
		#test_df.drop('surgicalcancerT',inplace = True, axis=1)
		#test_df.drop('surgicalcancerN',inplace = True, axis=1)
		#test_df.drop('surgicalcancerM',inplace = True, axis=1)
		#test_df.drop('boundarysurgical',inplace = True, axis=1)
		#test_df.drop('involvementrenal',inplace = True, axis=1)
		#test_df.drop('lymphrenal',inplace = True, axis=1)
		#test_df.drop('surgicalmethod',inplace = True, axis=1)
		test_df.drop('totaldose',inplace = True, axis=1)
		test_df.drop('radiationcnt',inplace = True, axis=1)
		test_df.drop('radiationperdose',inplace = True, axis=1)
		#test_df.drop('treatech',inplace = True, axis=1)
		#test_df.drop('ros1',inplace = True, axis=1)
		test_df.replace({'N': 0, 'Y': 1, 'Unknown' : 2, 'M' : 0, 'F' : 1}, inplace = True)



		test_df = test_df.astype('float32')
		test_x_minmax = scaler.transform(test_df)
		print(test_x_minmax.shape)

		cgb_final = GradientBoostingSurvivalAnalysis(n_estimators=FINAL_N, random_state=SEED)
		cgb_final.fit(train_x,train_y)
		print(cgb_final.score(val_x,val_y))
		surv = cgb_final.predict_survival_function(test_x_minmax , return_array=True)
		surv_df = pd.DataFrame(surv)
		surv_df = surv_df.transpose()

		ss = pd.DataFrame(index=list(range(MAX_DURATION+1)))
		ss = ss.join(surv_df)
		ss = ss.round(4)
		ss.fillna(method = 'bfill',inplace = True)
		ss.fillna(method = 'ffill',inplace = True)
		ss.rename(columns = lambda x: "ID_" + str(x) , inplace =True)
		predict_result_dir = os.path.join(PROJECT_DIR, 'results', 'predict')
		os.makedirs(predict_result_dir, exist_ok=True)
		ss.to_csv(os.path.join(predict_result_dir,MODEL_NAME+'.csv'), index_label = 'duration')
		print("Complete!!!~_~")







