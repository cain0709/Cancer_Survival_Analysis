""" 학습 코드
"""

import os, sys
import random
import argparse
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchtuples as tt
import torchtuples.callbacks as cb
from torch.utils.data import Dataset, DataLoader

from pycox.datasets import metabric
from pycox.models import CoxPH, PCHazard, DeepHitSingle, MTLR
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
MODEL_NAME = 'CoxPH_DNN_ver_30'

# Hyperparameters
SEED = 42
EPOCHS = 100
PATIENCE = 20
BS = 10000
LR = 1e-2
MAX_DURATION = 5500
LATENT_SIZE = 32
DROPOUT_RATIO = 0.5


class Net(nn.Module):
	def __init__(self,in_features):
		super(Net, self).__init__()
		self.dnn = nn.Sequential(
			nn.Linear(in_features,128),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(),
			nn.Dropout(DROPOUT_RATIO),
			nn.Linear(128,32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(),
			nn.Dropout(DROPOUT_RATIO),
			nn.Linear(32,16),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(),
			nn.Dropout(DROPOUT_RATIO),
			nn.Linear(16,1),
		)

		self.initialize_weights()

	def forward(self, x):
		x = self.dnn(x)
		return x

	def initialize_weights(self):
		# track all layers
		for m in self.modules():
			if isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)
				nn.init.constant_(m.bias, 0)
#Model

class Trainer():
	def __init__(self, model, train_loader, val_loader, scheduler, device,train_result_dir,val_x,val_y,train_x,train_y):
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.scheduler = scheduler
		self.device = device
		self.train_x = train_x
		self.train_y = train_y
		self.val_x = val_x
		self.val_y = val_y
		# Loss Function   

	def fit(self, ):
		best_c_index = 0
		#가장 높은 c-index
		best_loss = 1000000000 
		#가장 적은 loss값 기록
		#최종 임계값 저장 변수
		self.model._setup_train_info(self.train_loader)
		self.model.metrics = self.model._setup_metrics(None)
		self.model.val_metrics.dataloader = self.val_loader
		for epoch in range(EPOCHS):
			train_loss = []
			self.model.net.train()
			for data in self.train_loader:
				self.model.optimizer.zero_grad()
				self.model.batch_metrics = self.model.compute_metrics(data, self.model.metrics)
				self.model.batch_loss = self.model.batch_metrics["loss"]
				self.model.batch_loss.backward()
				self.model.optimizer.step()
				train_loss.append(self.model.batch_loss.item())
			
				#train
				
			mean_train_loss = np.mean(train_loss)
			c_index, val_loss = self.validation(self.model)
			#validation 수행
			
			print(f'Epoch : [{epoch+1}] Train loss : [{mean_train_loss}]\nVal c-index : [{c_index}])\nVal Loss : [{val_loss}]')
			self.scheduler.step(val_loss)
			#Scheduler 사용
			
			if best_loss > val_loss:
				best_loss = val_loss
				best_c_index = c_index
				self.model.save_net(os.path.join(train_result_dir,MODEL_NAME)) 
				print("Save Model~_~, MODEL_NAME : " + MODEL_NAME+" Criterion : Loss")

			'''if best_c_index < c_index:
				best_c_index = c_index
				best_loss = val_loss
				model.save_net(os.path.join(train_result_dir,MODEL_NAME)) #내장함수로 저장함
				print("Save Model~_~, MODEL_NAME : " + MODEL_NAME+ " Criterion : C-Index")'''
				

		print(f'Final Best Score : [{best_c_index}] ')
		print(f'Final Best Loss : [{best_loss}]')


	def validation(self, eval_model):
		val_loss = []
		eval_model.net.eval()
		with torch.no_grad():
			for data in self.val_loader:
				eval_model.batch_metrics = eval_model.compute_metrics(data, eval_model.metrics)
				eval_model.batch_loss = eval_model.score_in_batches(data)
				val_loss.append(eval_model.batch_loss['loss'])
				#val_loss.append(eval_model.batch_metrics['loss'].item())


		_ = eval_model.compute_baseline_hazards(input=self.train_x,target=self.train_y)
		surv = eval_model.predict_surv_df(self.val_x)
		ss = pd.DataFrame(index=list(range(int(max(surv.index) + 1))))
		ss = ss.join(surv)
		ss.fillna(method = 'bfill',inplace = True)
		ev = EvalSurv(ss, self.val_y[0], self.val_y[1], censor_surv='km')
		return ev.concordance_td(method='adj_antolini'), np.mean(val_loss)
	



def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
	os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
	torch.autograd.detect_anomaly(True)

	parser = argparse.ArgumentParser()
	KST = timezone(timedelta(hours=9))
	#parser.add_argument('--train_serial', type=str, default = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S"))
	parser.add_argument('--EDA', action='store_true')
	parser.add_argument('--Train', action='store_true')
	parser.add_argument('--Predict', action='store_true')
	#parser.add_argument('--train_serial', type=str, default = datetime.now().strftime("%Y%m%d_%H%M%S"))
	args=parser.parse_args()

	seed_everything(SEED)

	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Set train result directory
	train_result_dir = os.path.join(PROJECT_DIR, 'results', 'model')
	os.makedirs(train_result_dir, exist_ok=True)

	# Load data
	df = pd.read_csv(TRAIN_DATA_DIR)
	test_df = pd.read_csv(TEST_DATA_DIR)

	print(df)
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

		print(df.loc[df['sex'] == 'F','height'].describe())
		print(df.loc[df['sex'] == 'M','height'].describe())
		print(df.loc[df['sex'] == 'F','weight'].describe())
		print(df.loc[df['sex'] == 'M','weight'].describe())

		print(df.loc[df['sex'] == 'M','weight'].mean())
		print("###")
		print(df.loc[df['sex'] == 'F','weight'].mean())
		print("###")
		print(df.loc[df['sex'] == 'M','height'].mean())
		print("###")
		print(df.loc[df['sex'] == 'F','height'].mean())



		'''print(df.loc[df['totaldose'] == df['totaldose'], 'totaldose'])
		print(df.loc[df['radiationcnt'] == df['radiationcnt'], 'radiationcnt'])
		print(df.loc[df['radiationperdose'] == df['radiationperdose'], 'radiationperdose'])


		print(df.loc[df['totaldose'] == df['totaldose'], 'treatmethod'].value_counts())
		print(df.loc[df['radiationcnt'] == df['radiationcnt'], 'treatmethod'].value_counts())
		print(df.loc[df['radiationperdose'] == df['radiationperdose'], 'treatmethod'].value_counts())


		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 1), 'totaldose'].value_counts())
		print("Treatmethod 1 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 1), 'totaldose'].mean()))
		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 3), 'totaldose'].value_counts())
		print("Treatmethod 3 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 3), 'totaldose'].mean()))
		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 4), 'totaldose'].value_counts())
		print("Treatmethod 4 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 4), 'totaldose'].mean()))
		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 5), 'totaldose'].value_counts())
		print("Treatmethod 5 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 5), 'totaldose'].mean()))
		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 6), 'totaldose'].value_counts())
		print("Treatmethod 6 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 6), 'totaldose'].mean()))
		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 7), 'totaldose'].value_counts())
		print("Treatmethod 7 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 7), 'totaldose'].mean()))
		print(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 8), 'totaldose'].value_counts())
		print("Treatmethod 8 totaldose mean : " + str(df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 8), 'totaldose'].mean()))

		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 1), 'radiationcnt'].value_counts())
		print("Treatmethod 1 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 1), 'radiationcnt'].mean()))
		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 3), 'radiationcnt'].value_counts())
		print("Treatmethod 3 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 3), 'radiationcnt'].mean()))
		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 4), 'radiationcnt'].value_counts())
		print("Treatmethod 4 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 4), 'radiationcnt'].mean()))
		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 5), 'radiationcnt'].value_counts())
		print("Treatmethod 5 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 5), 'radiationcnt'].mean()))
		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 6), 'radiationcnt'].value_counts())
		print("Treatmethod 6 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 6), 'radiationcnt'].mean()))
		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 7), 'radiationcnt'].value_counts())
		print("Treatmethod 7 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 7), 'radiationcnt'].mean()))
		print(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 8), 'radiationcnt'].value_counts())
		print("Treatmethod 8 radiationcnt mean : " + str(df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 8), 'radiationcnt'].mean()))

		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 1), 'radiationperdose'].value_counts())
		print("Treatmethod 1 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 1), 'radiationperdose'].mean()))
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 3), 'radiationperdose'].value_counts())
		print("Treatmethod 3 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 3), 'radiationperdose'].mean()))
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 4), 'radiationperdose'].value_counts())
		print("Treatmethod 4 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 4), 'radiationperdose'].mean()))
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 5), 'radiationperdose'].value_counts())
		print("Treatmethod 5 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 5), 'radiationperdose'].mean()))
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 6), 'radiationperdose'].value_counts())
		print("Treatmethod 6 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 6), 'radiationperdose'].mean()))
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 7), 'radiationperdose'].value_counts())
		print("Treatmethod 7 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 7), 'radiationperdose'].mean()))
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 8), 'radiationperdose'].value_counts())
		print("Treatmethod 8 radiationperdose mean : " + str(df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 8), 'radiationperdose'].mean()))



		print(df.loc[df['totaldose'] == df['totaldose'], 'totaldose'].value_counts())
		print(df.loc[df['radiationcnt'] == df['radiationcnt'], 'treatmethod'].value_counts())
		print(df.loc[df['radiationperdose'] == df['radiationperdose'], 'treatmethod'].value_counts())

		print(df['treatmethod'].value_counts())
		print(df.loc[df['radiationperdose'] != df['radiationperdose'], 'treatmethod'].value_counts())''



		print(df.loc[(df['radiationperdose'] == df['radiationperdose']), 'cancerimagingT'].value_counts())
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']), 'cancerimagingN'].value_counts())
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']), 'cancerimagingM'].value_counts())


		print(df.loc[(df['radiationperdose'] == df['radiationperdose']), 'classificationcancer'].value_counts())
		print(df.loc[(df['radiationperdose'] == df['radiationperdose']), 'locationcancer'].value_counts())'''




		'''print("돌연빈이 value counts")
		print(df.loc[(df['egfr'] == 'N'), 'classificationcancer'].value_counts())
		print(df.loc[(df['egfr'] == 'Y'), 'classificationcancer'].value_counts())
		print(df.loc[(df['egfr'] != df['egfr']), 'classificationcancer'].value_counts())
		print(df.loc[(df['ros1'] == 'N'), 'classificationcancer'].value_counts())
		print(df.loc[(df['ros1'] == 'Y'), 'classificationcancer'].value_counts())
		print(df.loc[(df['ros1'] != df['ros1']), 'classificationcancer'].value_counts())
		print(df.loc[(df['alk'] == 'N'), 'classificationcancer'].value_counts())
		print(df.loc[(df['alk'] == 'Y'), 'classificationcancer'].value_counts())
		print(df.loc[(df['alk'] != df['alk']), 'classificationcancer'].value_counts())

		print(df.loc[(df['egfr'] == 'N') & (df['cancerimagingN'] != 0) , 'cancerimagingM'].value_counts())
		print(df.loc[(df['egfr'] == 'Y'), 'cancerimagingM'].value_counts())
		print(df.loc[(df['egfr'] != df['egfr']), 'cancerimagingM'].value_counts())
		print(df.loc[(df['ros1'] == 'N'), 'cancerimagingM'].value_counts())
		print(df.loc[(df['ros1'] == 'Y'), 'cancerimagingM'].value_counts())
		print(df.loc[(df['ros1'] != df['ros1']), 'cancerimagingM'].value_counts())
		print(df.loc[(df['alk'] == 'N'), 'cancerimagingM'].value_counts())
		print(df.loc[(df['alk'] == 'Y'), 'cancerimagingM'].value_counts())
		print(df.loc[(df['alk'] != df['alk']), 'cancerimagingM'].value_counts())

		print(df.loc[(df['relapse']==1), 'egfr'].value_counts())
		print(df.loc[(df['relapse']==1), 'ros1'].value_counts())
		print(df.loc[(df['relapse']==1), 'alk'].value_counts())

		print(df.loc[(df['relapse']==2), 'egfr'].value_counts())
		print(df.loc[(df['relapse']==2), 'ros1'].value_counts())
		print(df.loc[(df['relapse']==2), 'alk'].value_counts())

		print(df.loc[(df['relapse']==3), 'egfr'].value_counts())
		print(df.loc[(df['relapse']==3), 'ros1'].value_counts())
		print(df.loc[(df['relapse']==3), 'alk'].value_counts())'''

		print("Treatmethod 1")
		print(df.loc[df['treatmethod']==1,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==1,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==1,'alk'].value_counts())
		print("Treatmethod 2")
		print(df.loc[df['treatmethod']==2,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==2,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==2,'alk'].value_counts())
		print("Treatmethod 3")
		print(df.loc[df['treatmethod']==3,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==3,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==3,'alk'].value_counts())
		print("Treatmethod 4")
		print(df.loc[df['treatmethod']==4,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==4,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==4,'alk'].value_counts())
		print("Treatmethod 5")
		print(df.loc[df['treatmethod']==5,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==5,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==5,'alk'].value_counts())
		print("Treatmethod 6")
		print(df.loc[df['treatmethod']==6,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==6,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==6,'alk'].value_counts())
		print("Treatmethod 7")
		print(df.loc[df['treatmethod']==7,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==7,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==7,'alk'].value_counts())
		print("Treatmethod 8")
		print(df.loc[df['treatmethod']==8,'egfr'].value_counts())
		print(df.loc[df['treatmethod']==8,'ros1'].value_counts())
		print(df.loc[df['treatmethod']==8,'alk'].value_counts())

		print("Location 1")
		print(df.loc[df['locationcancer']==1,'egfr'].value_counts())
		print(df.loc[df['locationcancer']==1,'ros1'].value_counts())
		print(df.loc[df['locationcancer']==1,'alk'].value_counts())
		print("Location 2")
		print(df.loc[df['locationcancer']==2,'egfr'].value_counts())
		print(df.loc[df['locationcancer']==2,'ros1'].value_counts())
		print(df.loc[df['locationcancer']==2,'alk'].value_counts())
		print("Location 3")
		print(df.loc[df['locationcancer']==3,'egfr'].value_counts())
		print(df.loc[df['locationcancer']==3,'ros1'].value_counts())
		print(df.loc[df['locationcancer']==3,'alk'].value_counts())
		print("Location 4")
		print(df.loc[df['locationcancer']==4,'egfr'].value_counts())
		print(df.loc[df['locationcancer']==4,'ros1'].value_counts())
		print(df.loc[df['locationcancer']==4,'alk'].value_counts())
		print("Location 5")
		print(df.loc[df['locationcancer']==5,'egfr'].value_counts())
		print(df.loc[df['locationcancer']==5,'ros1'].value_counts())
		print(df.loc[df['locationcancer']==5,'alk'].value_counts())
		print("Location 9")
		print(df.loc[df['locationcancer']==9,'egfr'].value_counts())
		print(df.loc[df['locationcancer']==9,'ros1'].value_counts())
		print(df.loc[df['locationcancer']==9,'alk'].value_counts())


		print("Cancerimaging T, N, M")
		print(df['cancerimagingT'].value_counts())
		print(df['cancerimagingN'].value_counts())
		print(df['cancerimagingM'].value_counts())

		print(df.loc[(df['cancerimagingM'] == 1), 'cancerimagingT'].value_counts())
		print(df.loc[(df['cancerimagingM'] == 1), 'cancerimagingN'].value_counts())
		print(df.loc[(df['cancerimagingT'] == 0) & (df['cancerimagingM'] == 1), 'cancerimagingN'].value_counts())
		print(df.loc[(df['cancerimagingN'] == 0) & (df['cancerimagingM'] == 1), 'cancerimagingT'].value_counts())

		print(df.loc[df['deathsign']!=df['deathsign'], 'cancerimagingM'].value_counts())
		print(df.loc[df['deathsign']==1, 'cancerimagingM'].value_counts())
		print(df.loc[df['deathsign']==2, 'cancerimagingM'].value_counts())

		print(df.loc[(df['deathsign']!=df['deathsign']), 'dead'].value_counts())




		'''print("###")
		print(df.loc[df['sex'] == 'M','weight'].median())
		print("###")
		print(df.loc[df['sex'] == 'F','weight'].median())
		print("###")
		print(df.loc[df['sex'] == 'M','height'].median())
		print("###")
		print(df.loc[df['sex'] == 'F','height'].median())'''

	#scaler
	scaler = MinMaxScaler()


	#preprocess
	#weight_mean = df['weight'].mean()
	#height_mean = df['height'].mean()

	weight_mean_m = df.loc[df['sex'] == 'M','weight'].mean()
	weight_mean_f = df.loc[df['sex'] == 'F','weight'].mean()
	height_mean_m = df.loc[df['sex'] == 'M','height'].mean()
	height_mean_f = df.loc[df['sex'] == 'F','height'].mean()
	

	totaldose_mean_1 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 1), 'totaldose'].mean()
	totaldose_mean_3 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 3), 'totaldose'].mean()
	totaldose_mean_4 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 4), 'totaldose'].mean()
	totaldose_mean_5 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 5), 'totaldose'].mean()
	totaldose_mean_6 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 6), 'totaldose'].mean()
	totaldose_mean_7 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 7), 'totaldose'].mean()
	totaldose_mean_8 = df.loc[(df['totaldose'] == df['totaldose']) & (df['treatmethod'] == 8), 'totaldose'].mean()

	radiationcnt_mean_1 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 1), 'radiationcnt'].mean()
	radiationcnt_mean_3 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 3), 'radiationcnt'].mean()
	radiationcnt_mean_4 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 4), 'radiationcnt'].mean()
	radiationcnt_mean_5 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 5), 'radiationcnt'].mean()
	radiationcnt_mean_6 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 6), 'radiationcnt'].mean()
	radiationcnt_mean_7 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 7), 'radiationcnt'].mean()
	radiationcnt_mean_8 = df.loc[(df['radiationcnt'] == df['radiationcnt']) & (df['treatmethod'] == 8), 'radiationcnt'].mean()

	radiationperdose_mean_1 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 1), 'radiationperdose'].mean()
	radiationperdose_mean_3 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 3), 'radiationperdose'].mean()
	radiationperdose_mean_4 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 4), 'radiationperdose'].mean()
	radiationperdose_mean_5 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 5), 'radiationperdose'].mean()
	radiationperdose_mean_6 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 6), 'radiationperdose'].mean()
	radiationperdose_mean_7 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 7), 'radiationperdose'].mean()
	radiationperdose_mean_8 = df.loc[(df['radiationperdose'] == df['radiationperdose']) & (df['treatmethod'] == 8), 'radiationperdose'].mean()


	totaldose_mean = df['totaldose'].mean()
	radiationcnt_mean = df['radiationcnt'].mean()
	radiationperdose_mean = df['radiationperdose'].mean()

	df.loc[(df['weight'] != df['weight']) & (df['sex'] == 'M'), 'weight'] = weight_mean_m 
	df.loc[(df['weight'] != df['weight']) & (df['sex'] == 'F'), 'weight'] = weight_mean_f
	df.loc[(df['height'] != df['height']) & (df['sex'] == 'M'), 'height'] = height_mean_m
	df.loc[(df['height'] != df['height']) & (df['sex'] == 'F'), 'height'] = height_mean_f

	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 1), 'totaldose'] = totaldose_mean_1
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 2), 'totaldose'] = 0
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 3), 'totaldose'] = totaldose_mean_3
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 4), 'totaldose'] = totaldose_mean_4
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 5), 'totaldose'] = totaldose_mean_5
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 6), 'totaldose'] = totaldose_mean_6
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 7), 'totaldose'] = totaldose_mean_7
	df.loc[(df['totaldose'] != df['totaldose']) & (df['treatmethod'] == 8), 'totaldose'] = totaldose_mean_8
	df.loc[df['treatmethod'] != df['treatmethod'], 'totaldose'] = totaldose_mean

	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 1), 'radiationcnt'] = radiationcnt_mean_1
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 2), 'radiationcnt'] = 0
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 3), 'radiationcnt'] = radiationcnt_mean_3
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 4), 'radiationcnt'] = radiationcnt_mean_4
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 5), 'radiationcnt'] = radiationcnt_mean_5
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 6), 'radiationcnt'] = radiationcnt_mean_6
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 7), 'radiationcnt'] = radiationcnt_mean_7
	df.loc[(df['radiationcnt'] != df['radiationcnt']) & (df['treatmethod'] == 8), 'radiationcnt'] = radiationcnt_mean_8
	df.loc[df['treatmethod'] != df['treatmethod'], 'radiationcnt'] = radiationcnt_mean

	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 1), 'radiationperdose'] = radiationperdose_mean_1
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 2), 'radiationperdose'] = 0
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 3), 'radiationperdose'] = radiationperdose_mean_3
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 4), 'radiationperdose'] = radiationperdose_mean_4
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 5), 'radiationperdose'] = radiationperdose_mean_5
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 6), 'radiationperdose'] = radiationperdose_mean_6
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 7), 'radiationperdose'] = radiationperdose_mean_7
	df.loc[(df['radiationperdose'] != df['radiationperdose']) & (df['treatmethod'] == 8), 'radiationperdose'] = radiationperdose_mean_8
	df.loc[df['treatmethod'] != df['treatmethod'], 'radiationperdose'] = radiationperdose_mean

	#df['weight'].fillna(weight_mean, inplace = True)
	#df['height'].fillna(height_mean, inplace = True)
	df['familyhistory'].fillna(0 ,inplace = True)
	df['bp'].fillna(0,inplace = True) 
	df['bs'].fillna(0,inplace = True)
	df['sm'].fillna(0,inplace = True)
	df['cancerimagingN'].fillna(0,inplace = True)
	df['deathsign'].fillna(0,inplace = True)
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
	#df['totaldose'].fillna(totaldose_mean, inplace = True)
	#df['radiationcnt'].fillna(radiationcnt_mean, inplace = True)
	#df['radiationperdose'].fillna(radiationperdose_mean, inplace = True)

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
	#df.drop('totaldose',inplace = True, axis=1)
	#df.drop('radiationcnt',inplace = True, axis=1)
	#df.drop('radiationperdose',inplace = True, axis=1)
	#df.drop('treatech',inplace = True, axis=1)
	#df.drop('ros1',inplace = True, axis=1)


	onehot_cols =['locationcancer','classificationcancer','surgicalcancerT','surgicalcancerN',
	'surgicalcancerM','boundarysurgical','involvementrenal','lymphrenal','surgicalmethod','treatmethod','treatech','egfr','ros1','alk']


	#df = pd.get_dummies(df, columns=onehot_cols)
	
	print(df.isnull().sum())
	print(df.shape)
	print(df.columns)

	df.replace({'N': 0, 'Y': 1, 'Unknown' : 2, 'M' : 0, 'F' : 1}, inplace = True)

	#onehot_cols =['locationcancer','classificationcancer','surgicalmethod','treatmethod','treatech']

	df = pd.get_dummies(df, columns=onehot_cols)

	col = list(df.columns)
	col.remove('duration')
	col.remove('dead')

	df_train, df_val = train_test_split(df, test_size = 0.05, stratify = df['dead'], random_state=SEED)

	get_target = lambda df: (df['duration'].values, df['dead'].values)
	df_train = df_train.astype('float32')
	df_val = df_val.astype('float32')


	train_y = get_target(df_train)
	val_y = get_target(df_val)

	df_train.drop(['duration','dead'],axis=1,inplace=True)
	df_val.drop(['duration','dead'],axis=1,inplace=True)

	'''#deathsign labeling

	ds_train_y = df_train['deathsign']
	ds_val_y = df_val['deathsign']
	
	ds_train_x = df_train.drop(['deathsign'],axis=1)
	ds_val_x = df_val.drop(['deathsign'],axis=1)
	print(ds_train_x.columns)
	print(ds_val_x.columns)
	ds_train_x = scaler.fit_transform(ds_train_x)
	ds_val_x = scaler.transform(ds_val_x)


	for i in range(30):
		n = 10 + 10*i
		classifier = RandomForestClassifier(n_estimators = n, min_samples_split =10, criterion = 'entropy', random_state = SEED)
		classifier.fit(ds_train_x, ds_train_y)
		ds_train_pred =classifier.predict(ds_train_x)
		ds_val_pred = classifier.predict(ds_val_x)
		print("N : "+str(n))
		print(confusion_matrix(ds_train_y,ds_train_pred))
		print(confusion_matrix(ds_val_y,ds_val_pred))
		print(classification_report(ds_val_y, ds_val_pred))


	
	clf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = SEED)
	clf.fit(ds_train_x, ds_train_y)
	df_train['deathsign'] = clf.predict(ds_train_x)
	df_val['deathsign'] = clf.predict(ds_val_x)

	###'''

	print(df_train.columns)
	print(df_val.columns)

	train_x = scaler.fit_transform(df_train)
	val_x = scaler.transform(df_val)


	print("Train shape")
	print(train_x.shape)
	print("Val shape")
	print(val_x.shape)



	#train part
	if args.Train is True:
		#train_loader, val_loader 선언
		#train_dataset = MyDataset(train_x,train_y,test_mode=False)
		#train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
		#val_dataset = MyDataset(val_x,val_y,test_mode=False)
		#val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)

		
		
		'''#Autoencoder
		AE = Autoencoder(in_features)
		optimizer_ae = torch.optim.AdamW(params = AE.parameters(), lr = LR,weight_decay=1e-4)#Opimizer : Adam
		scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.5, patience=20, threshold_mode='abs', min_lr=1e-8, verbose=True)
		train_dataset = MyDataset(train_x)
		train_loader_ae = DataLoader(train_dataset, batch_size=BS, shuffle=True)
		val_dataset = MyDataset(val_x)
		val_loader_ae = DataLoader(val_dataset, batch_size=BS, shuffle=False)
		#train_loader, val_loader 선언
		trainer_ae = Trainer_AE(AE, optimizer_ae, train_loader_ae, val_loader_ae, scheduler_ae, device)
		trainer_ae.fit()'''


		'''pretrain_AE = torch.load(os.path.join(train_result_dir,"Autoencoder.pth"))
		pretrain_AE.eval()
		tensor_train = torch.tensor(train_x).to(device)
		tensor_val = torch.tensor(val_x).to(device)
		train_x, _  = pretrain_AE(tensor_train)
		val_x, _  = pretrain_AE(tensor_val)
		train_x = train_x.cpu()
		val_x = val_x.cpu()
		train_x = train_x.detach().numpy()
		val_x = val_x.detach().numpy()

		print(train_x.shape)
		print(val_x)'''

		in_features = train_x.shape[1]
		net = Net(in_features)

	
		optimizer = torch.optim.AdamW(params = net.parameters(), lr = LR, weight_decay=1e-2)#Opimizer : AdamW
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE, threshold_mode='abs', min_lr=1e-6, verbose=True)
		## Training the model
		seed_everything(SEED) # Seed 고정 (모델 생성전에 다시 고정하였더니 재현이 똑같이됨..)


		model = CoxPH(net, optimizer, device)
	

		train_input = (train_x, train_y)
		#print(train_x_input)
		val_input = val_x, val_y
		#print(val_x_input)
		train_loader = model.make_dataloader(train_input, BS ,shuffle = True, num_workers=0)
		val_loader = model.make_dataloader(val_input, BS, shuffle = False, num_workers=0)
		trainer = Trainer(model, train_loader, val_loader, scheduler, device, train_result_dir, val_x, val_y,train_x,train_y)
		trainer.fit()

		# Train
		net = torch.load(os.path.join(train_result_dir,MODEL_NAME+'.pt'))
		print("Load model : " + MODEL_NAME)
		net.eval()
		pretrain_model = CoxPH(net, torch.optim.AdamW(params=net.parameters(),lr=LR,weight_decay=1e-4), device)
		pretrain_model.load_net(os.path.join(train_result_dir,MODEL_NAME))

		surv = pretrain_model.predict_surv_df(val_x)

		ss = pd.DataFrame(index=list(range(int(max(surv.index) + 1))))
		ss = ss.join(surv)
		ss.fillna(method = 'bfill',inplace = True)

		# y_val[0]: duration , y_val[1]: event

		ev = EvalSurv(ss, val_y[0], val_y[1], censor_surv='km')

		print("c-index : " , ev.concordance_td(method='adj_antolini'))


	if args.Predict is True:
		test_df = pd.read_csv(TEST_DATA_DIR)
		
		#성별마다 다르게 몸무게 및 키 대치
		test_df.loc[(test_df['weight'] != test_df['weight']) & (test_df['sex'] == 'M'), 'weight'] = weight_mean_m 
		test_df.loc[(test_df['weight'] != test_df['weight']) & (test_df['sex'] == 'F'), 'weight'] = weight_mean_f
		test_df.loc[(test_df['height'] != test_df['height']) & (test_df['sex'] == 'M'), 'height'] = height_mean_m
		test_df.loc[(test_df['height'] != test_df['height']) & (test_df['sex'] == 'F'), 'height'] = height_mean_f

		#방사선 양을 수술기법에 따라 다르게 대
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 1), 'totaldose'] = totaldose_mean_1
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 2), 'totaldose'] = 0
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 3), 'totaldose'] = totaldose_mean_3
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 4), 'totaldose'] = totaldose_mean_4
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 5), 'totaldose'] = totaldose_mean_5
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 6), 'totaldose'] = totaldose_mean_6
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 7), 'totaldose'] = totaldose_mean_7
		test_df.loc[(test_df['totaldose'] != test_df['totaldose']) & (test_df['treatmethod'] == 8), 'totaldose'] = totaldose_mean_8
		test_df.loc[test_df['treatmethod'] != test_df['treatmethod'], 'totaldose'] = totaldose_mean

		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 1), 'radiationcnt'] = radiationcnt_mean_1
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 2), 'radiationcnt'] = 0
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 3), 'radiationcnt'] = radiationcnt_mean_3
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 4), 'radiationcnt'] = radiationcnt_mean_4
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 5), 'radiationcnt'] = radiationcnt_mean_5
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 6), 'radiationcnt'] = radiationcnt_mean_6
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 7), 'radiationcnt'] = radiationcnt_mean_7
		test_df.loc[(test_df['radiationcnt'] != test_df['radiationcnt']) & (test_df['treatmethod'] == 8), 'radiationcnt'] = radiationcnt_mean_8
		test_df.loc[test_df['treatmethod'] != test_df['treatmethod'], 'radiationcnt'] = radiationcnt_mean

		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 1), 'radiationperdose'] = radiationperdose_mean_1
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 2), 'radiationperdose'] = 0
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 3), 'radiationperdose'] = radiationperdose_mean_3
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 4), 'radiationperdose'] = radiationperdose_mean_4
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 5), 'radiationperdose'] = radiationperdose_mean_5
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 6), 'radiationperdose'] = radiationperdose_mean_6
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 7), 'radiationperdose'] = radiationperdose_mean_7
		test_df.loc[(test_df['radiationperdose'] != test_df['radiationperdose']) & (test_df['treatmethod'] == 8), 'radiationperdose'] = radiationperdose_mean_8
		test_df.loc[test_df['treatmethod'] != test_df['treatmethod'], 'radiationperdose'] = radiationperdose_mean


		test_df['imagelevel'] = ''

		test_df['cancerimagingN'].fillna(0,inplace = True)

		test_df.loc[(test_df['cancerimagingT'] == 0), 'imagelevel'] = 0
		test_df.loc[(test_df['cancerimagingT'] == 1) & (test_df['cancerimagingN'] == 0), 'imagelevel'] = 1
		test_df.loc[(test_df['cancerimagingT'] == 2) & (test_df['cancerimagingN'] == 0), 'imagelevel'] = 2
		test_df.loc[(test_df['cancerimagingT'] == 1) & (test_df['cancerimagingN'] == 1), 'imagelevel'] = 3
		test_df.loc[(test_df['cancerimagingT'] == 2) & (test_df['cancerimagingN'] == 1), 'imagelevel'] = 4
		test_df.loc[(test_df['cancerimagingT'] == 3) & (test_df['cancerimagingN'] == 0), 'imagelevel'] = 4
		test_df.loc[(test_df['cancerimagingT'] == 3) & (test_df['cancerimagingN'] == 1), 'imagelevel'] = 5
		test_df.loc[(test_df['cancerimagingT'] == 1) & (test_df['cancerimagingN'] == 2), 'imagelevel'] = 5
		test_df.loc[(test_df['cancerimagingT'] == 2) & (test_df['cancerimagingN'] == 2), 'imagelevel'] = 5
		test_df.loc[(test_df['cancerimagingT'] == 3) & (test_df['cancerimagingN'] == 2), 'imagelevel'] = 5

		test_df.loc[(test_df['cancerimagingT'] == 4) & (test_df['cancerimagingN'] == 0), 'imagelevel'] = 6
		test_df.loc[(test_df['cancerimagingT'] == 4) & (test_df['cancerimagingN'] == 1), 'imagelevel'] = 6
		test_df.loc[(test_df['cancerimagingT'] == 4) & (test_df['cancerimagingN'] == 2), 'imagelevel'] = 6
		test_df.loc[(test_df['cancerimagingT'] == 1) & (test_df['cancerimagingN'] == 3), 'imagelevel'] = 6
		test_df.loc[(test_df['cancerimagingT'] == 2) & (test_df['cancerimagingN'] == 3), 'imagelevel'] = 6
		test_df.loc[(test_df['cancerimagingT'] == 3) & (test_df['cancerimagingN'] == 3), 'imagelevel'] = 6
		test_df.loc[(test_df['cancerimagingT'] == 4) & (test_df['cancerimagingN'] == 3), 'imagelevel'] = 6

		test_df.loc[test_df['cancerimagingM'] == 1, 'imagelevel'] = 7

		test_df['familyhistory'].fillna("Unknown" ,inplace = True)
		test_df['bp'].fillna("Unknown",inplace = True) 
		test_df['bs'].fillna("Unknown",inplace = True)
		test_df['sm'].fillna("Unknown",inplace = True)
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
		#test_df['totaldose'].fillna(totaldose_mean, inplace = True)
		#test_df['radiationcnt'].fillna(radiationcnt_mean, inplace = True)
		#test_df['radiationperdose'].fillna(radiationperdose_mean, inplace = True)


		#df.drop('surgicalcancerT',inplace = True, axis=1)
		#df.drop('surgicalcancerN',inplace = True, axis=1)
		#df.drop('surgicalcancerM',inplace = True, axis=1)
		#df.drop('boundarysurgical',inplace = True, axis=1)
		#df.drop('involvementrenal',inplace = True, axis=1)
		#df.drop('lymphrenal',inplace = True, axis=1)
		#df.drop('surgicalmethod',inplace = True, axis=1)


		test_df['BMI'] = test_df['weight'] / (test_df['height']*test_df['height'])


		test_df.drop('weight',inplace=True,axis=1)
		test_df.drop('height',inplace=True,axis=1)
		test_df.drop('cancerimagingT',inplace=True,axis=1)
		test_df.drop('cancerimagingN',inplace=True,axis=1)
		test_df.drop('cancerimagingM',inplace=True,axis=1)
		test_df.drop('surgicalcancerT',inplace=True,axis=1)
		test_df.drop('surgicalcancerN',inplace=True,axis=1)
		test_df.drop('surgicalcancerM',inplace=True,axis=1)

		#test_df.drop('totaldose',inplace = True, axis=1)
		#test_df.drop('radiationperdose',inplace = True, axis=1)
		#df.drop('treatech',inplace = True, axis=1)
		#df.drop('ros1',inplace = True, axis=1)

		test_df.replace({'N': 0, 'Y': 1, 'Unknown' : 2, 'M' : 0, 'F' : 1}, inplace = True)

		test_df = pd.get_dummies(test_df, columns=onehot_cols)
		print(test_df.isnull().sum())
		print(test_df.shape)
		print(test_df.columns)


		test_df = test_df.astype('float32')
		test_x_minmax = scaler.transform(test_df)

		print(test_x_minmax.shape)
		net = torch.load(os.path.join(train_result_dir,MODEL_NAME+'.pt'))
		print("Load model : " + MODEL_NAME)
		net.eval()
		pretrain_model = CoxPH(net, torch.optim.AdamW(params=net.parameters(),lr=LR,weight_decay=1e-2), device)
		pretrain_model.load_net(os.path.join(train_result_dir,MODEL_NAME))



		surv = pretrain_model.predict_surv_df(val_x)
		ss = pd.DataFrame(index=list(range(int(max(surv.index) + 1))))
		ss = ss.join(surv)
		ss.fillna(method = 'bfill',inplace = True)
		ev = EvalSurv(ss, val_y[0], val_y[1], censor_surv='km')
		print("C-index : " + str(ev.concordance_td(method='adj_antolini')))



		surv = pretrain_model.predict_surv_df(test_x_minmax)
		print(surv.shape)
		ss = pd.DataFrame(index=list(range(MAX_DURATION+1)))
		ss = ss.join(surv)
		ss = ss.round(4)
		ss.fillna(method = 'bfill',inplace = True)
		ss.fillna(method = 'ffill',inplace = True)
		ss.rename(columns = lambda x: "ID_" + str(x) , inplace =True)
		predict_result_dir = os.path.join(PROJECT_DIR, 'results', 'predict')
		os.makedirs(predict_result_dir, exist_ok=True)
		ss.to_csv(os.path.join(predict_result_dir,MODEL_NAME+'.csv'), index_label = 'duration')
		print("Complete!!!~_~")







