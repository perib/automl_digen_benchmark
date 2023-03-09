import os
import tpot
import h2o
import pickle
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import make_scorer

from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, \
                             f1_score)

savepath = "/home/ribeirop/common/Projects/tpot_digen_paper1/AutoML_Results_HPC"


automl_experiments = [

                        {
                    'automl': 'h2o',
                    'exp_name' : 'h2o_1200s',
                    },


                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_STC_1200s',
                    
                    },

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_C_1200s',

                    },

                    

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_base_1200s',
                    
                    },

                                                            {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_STC_3600s',
                    
                    },

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_C_3600s',

                    },

                    

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_base_3600s',
                    
                    },

                    {
                    'automl': 'h2o',
                    'exp_name' : 'h2o_3600s',
                    },


        ]

bigger_testset = pickle.load(open("all_new_test_sets.pkl", "rb"))

all_results = []
for exp in automl_experiments:
    print(exp)
    for run in range(0,10):
        if not os.path.exists(os.path.join(savepath,"digen_results",exp['automl'],exp['exp_name'],f"results_run_{run}")):
            continue
        est = pickle.load(open(os.path.join(savepath,"digen_results",exp['automl'],exp['exp_name'],f"results_run_{run}","estimators.pkl"),"rb"))

        for dset in bigger_testset.keys():
            print(dset)
            sys.stdout.flush()
            
            
            X = bigger_testset[dset]['X']
            y = bigger_testset[dset]['y']
            y_no_noise = bigger_testset[dset]['y_no_noise']

            if exp['automl'] == 'h2o':
                h2o.init(   nthreads=48, max_mem_size ="1000G")
                h2o_model = h2o.load_model(est[dset]['classifier'])
                all_y_preds = h2o_model.predict(h2o.H2OFrame(X.to_numpy()))
               
                y_preds = all_y_preds[0].as_data_frame().to_numpy()
                y_proba = all_y_preds[2].as_data_frame().to_numpy()
                y_des = y_proba
            
            else:
                y_preds = est[dset]['classifier'].predict(X)
                if hasattr(est[dset]['classifier'], "predict_proba"):
                    y_proba = est[dset]['classifier'].predict_proba(X)[:,1]
                else: 
                    y_proba = y_preds
                
                if hasattr(est[dset]['classifier'], "decision_function"):
                    y_des = est[dset]['classifier'].decision_function(X)
                else: 
                    y_des = y_proba


            this_auroc_score = roc_auc_score(y, y_des)
            this_accuracy_score = accuracy_score(y, y_preds)
            this_balanced_accuracy_score = balanced_accuracy_score(y, y_preds)
            this_f1_score = f1_score(y, y_preds)

            this_auroc_score_y_no_noise = roc_auc_score(y_no_noise, y_des)
            this_accuracy_score_y_no_noise = accuracy_score(y_no_noise, y_preds)
            this_balanced_accuracy_score_y_no_noise = balanced_accuracy_score(y_no_noise, y_preds)
            this_f1_score_y_no_noise = f1_score(y_no_noise, y_preds)

            prec, rec, _ = precision_recall_curve(y, y_des)
            prec_no_noise, rec_no_noise, _ = precision_recall_curve(y_no_noise, y_des)

            print(this_auroc_score)
            all_results.append({'dataset':dset, 
                                
                                'auroc':this_auroc_score,
                                'accuracy':this_accuracy_score,
                                'balanced_accuracy':this_balanced_accuracy_score,
                                'f1_score':this_f1_score,
                                'prec': prec,
                                'rec': rec,
                                'auprc': auc(rec, prec),

                                'auroc_y_no_noise':this_auroc_score_y_no_noise,
                                'accuracy_y_no_noise':this_accuracy_score_y_no_noise,
                                'balanced_accuracy_y_no_noise':this_balanced_accuracy_score_y_no_noise,
                                'f1_score_y_no_noise':this_f1_score_y_no_noise,
                                'prec_y_no_noise': prec_no_noise,
                                'rec_y_no_noise': rec_no_noise,
                                'auprc_y_no_noise': auc(rec_no_noise, prec_no_noise),

                                'run':run, 
                                'automl':exp['automl'], 
                                'exp_name':exp['exp_name'],
                                'y_preds':y_preds,
                                'y_proba':y_proba,
                                'y_des':y_des,
                                
                                })



df = pd.DataFrame(all_results)
df.to_csv("tpot_retest_results.csv")