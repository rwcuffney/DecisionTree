import pandas as pd
import streamlit as st

import matplotlib.pyplot  as plt
import category_encoders as ce
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn import metrics

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

#models:
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier



def test_fuction():
    st.write('this is only a test')

def page_header(general_name, model,**kwargs):
    model_name = type(model).__name__
    st.title(general_name)
    st.header(f'model: {model_name}')
    st.write(f"let's take a look at how {general_name} performs on the dataset.")

    # Create the button
    if st.button('Click me to run the model',key='RunCode_button'):
        #dataset = myData()
        df = data_cleanup(**kwargs)
        st.write('This is how the model was set-up:')
        st.code(model)
        split_scale_fit(model=model,df=df,**kwargs)
        graph_ROC_AUC(model=model,df=df,**kwargs)

        #st.header('Balance the dataset')
        #df,X,y = data_cleanup(Fraud=5_000,Non_Fraud=90_000)
        #split_scale_fit(model=model,X=X,y=y,test_size=.3)
        #graph_ROC_AUC(model=model,X=X,y=y,test_size=.3)

@st.cache_data
def myData():
    dataset1 = pd.read_csv('./data/fraudTest.csv')
    dataset2 = pd.read_csv('./data/fraudTrain.csv')
    frames = [dataset1,dataset2]
    dataset = pd.concat(frames,ignore_index=True)
    return dataset

def metrics_to_csv(model_name,score_list):
    file_name = Path('./data/Metrics.csv')
    df_Metrics=pd.read_csv(file_name, index_col=0)
    df_Metrics[model_name] = score_list
    df_Metrics.to_csv(file_name)

def o_encode(cat,df):
    '''
    let’s convert these categories from text to numbers. 
    For this, we can use Scikit-Learn’s 
    OrdinalEncoder class
    '''
    from sklearn.preprocessing import OrdinalEncoder

    #categories = ['Brand','Model','PowerTrain','BodyStyle','Segment','p_type']

    for c in cat:
        #u = len(list(set(list(df[c].tolist()))))
        #print(u,c)
        ordinal_encoder = OrdinalEncoder()
        #print(df[c].head())
        #housing_cat_encoded = ordinal_encoder.fit_transform(df)
        df_cat_encoded = ordinal_encoder.fit_transform(df[[c]])
        df[c] = df_cat_encoded 
    df = df.apply(pd.to_numeric, args=('coerce',))
    return df


def data_cleanup(**kwargs):
    df = myData()
    #look for numerical columns and only keep those:
    
    #encoder = ce.OneHotEncoder(cols=['category'])
    #df = encoder.fit_transform(df)


    categories=['merchant','category','state','job']
    df = o_encode(categories,df)

    
    Numerical = [col for col in df.columns if df[col].dtypes != 'O']
    df =df[Numerical]


    my_cols = [col for col in df.columns if df[col].isnull().sum() == 0]
    #look for columns with zero null values, and only keep those:
    df=df[my_cols]
    
    #drop Unnamed column:
    df=df.drop(['Unnamed: 0'], axis=1)
    
    #This seemed to make matters worse:
    #df = df[['amt','zip','merch_lat','merch_long','is_fraud']]
    ### Use one hot encoder to set binary fields.
    #df = data_cleanup()

    if 'Fraud' in kwargs:
        df = balanced_dataframe(df,kwargs['Fraud'],kwargs['Non_Fraud'])
    X,y = get_X_y(df)
    
    return df





def balanced_dataframe(df,Fraud,Non_Fraud):
    non_fraud = df[df['is_fraud']==0]
    fraud = df[df['is_fraud']==1]
    non_fraud = non_fraud.sample(frac=1)
    non_fraud = non_fraud[:Non_Fraud]
    new_df = pd.concat([non_fraud,fraud.sample(Fraud)])
    balanced_df = new_df.sample(frac=1)
    st.write('This model is run from a balanced dataset')
    st.write(f'This dataset is balanced to: [Fraud:{Fraud}, Non_Fraud:{Non_Fraud}')
    return  balanced_df

def get_X_y(df):
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud'].to_frame()
    return X,y


def show_metrics(model, test, pred):
    my_precision_score = metrics.precision_score(test, pred, pos_label=1)
    st.write(f'Precision = {my_precision_score:.2%}')
    my_recall_score = metrics.recall_score(test, pred, pos_label=1)
    st.write(f'Recall = {my_recall_score:.2%}')
    
    
    model_name = type(model).__name__
    fn = Path(f'output/{model_name}_Confusion_Matrix')
    #CONFUSION MATRIX
    labels = ['No Fraud','Fraud']
    cfm= metrics.confusion_matrix(test, pred,labels=model.classes_)
    st.write(f'confusion_matrix:\n',cfm)
    

    #CLASSIFICATION REPORT
    #st.write(metrics.classification_report(test, pred))
    report =metrics.classification_report(test, pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().head(2)	
    st.dataframe(df_report )
    
    ###DISPLAY CONFUSION MATRIX
    fig, ax = plt.subplots(figsize=(5,5))
    plt.rcParams.update({'font.size': 22})
    cmd = metrics.ConfusionMatrixDisplay(cfm,display_labels=labels)
    
    ax = cmd.plot(ax=ax, 
                  colorbar=False,
                  values_format = '.0f',
                  cmap='tab20')#'Accent') see color options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.savefig(fn,bbox_inches='tight')
    st.pyplot(fig)

    my_Accuracy_score = metrics.accuracy_score(test, pred)
    my_Weighted_f1 = metrics.f1_score(test, pred,average='weighted')
    my_Cohen_Kappa = metrics.cohen_kappa_score(test, pred)
    my_Matthews_coefficient = metrics.matthews_corrcoef(test, pred)
      
    #ACCURACY SCORES:
    st.write(f"Accuracy Score: {my_Accuracy_score:.4f}")
    #st.write(f"ROC AUC score: {roc_auc_score(y, model.predict_proba(X)):.4f}")
    st.write(f"Weighted f1 score: {my_Weighted_f1:.4f}")
    st.write(f'Cohen Kappa score: {my_Cohen_Kappa:.4f}')
    st.write(f'Matthews coefficient: {my_Matthews_coefficient:.4f}')


    score_list = [my_Accuracy_score, my_precision_score, my_recall_score, my_Weighted_f1, my_Cohen_Kappa, my_Matthews_coefficient]

    metrics_to_csv(model_name,score_list)
    
# def scale(scaler,*args):
#     lst=[]
#     for a in args:
#         pd.DataFrame(scaler.fit_transform(a))
#         lst.append(a)
#     return tuple(lst)
    
def split_scale_fit(model,df,**kwargs):
    if 'fraud_test_size' in kwargs:
        fraud_test_size=kwargs['fraud_test_size']
        non_fraud_test_size=kwargs['non_fraud_test_size']

        non_fraud = df[df['is_fraud']==0]
        fraud = df[df['is_fraud']==1]

        X,y = get_X_y(fraud)
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X, y, test_size = fraud_test_size, random_state = 0)

        X,y = get_X_y(non_fraud)
        X_train_non_fraud, X_test_non_fraud, y_train_non_fraud, y_test_non_fraud = train_test_split(X, y, test_size = non_fraud_test_size, random_state = 0)

        print(f'Training set balance= [Fraud:{len(X_train_fraud)}, Non_Fraud:{len(X_train_non_fraud)}]')
        print(f'Testing set balance= [Fraud:{len(X_test_fraud)}, Non_Fraud:{len(X_test_non_fraud)}]')

        X_train= pd.concat([X_train_fraud,X_train_non_fraud])
        X_test= pd.concat([X_test_fraud,X_test_non_fraud])
        y_train= pd.concat([y_train_fraud,y_train_non_fraud])
        y_test= pd.concat([y_test_fraud,y_test_non_fraud])   
    else:
        X,y = get_X_y(df)
        test_size=kwargs['test_size']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    
    #scale X datasets:
    
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    
    #fit the model:
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    

    auc = metrics.accuracy_score(y_test, y_pred)
    st.write(auc)
    show_metrics(model, y_test, y_pred)
    
    #st.write('********* COMPARE MODEL ON ENTIRE DATASET.  *********')
    #X = pd.DataFrame(scaler.fit_transform(X))
    #y_pred_entiredataset = model.predict(X)
    #show_metrics(model,y, y_pred_entiredataset)
    
def graph_ROC_AUC(model,df,**kwargs):
    if 'fraud_test_size' in kwargs:
        fraud_test_size=kwargs['fraud_test_size']
        non_fraud_test_size=kwargs['non_fraud_test_size']

        non_fraud = df[df['is_fraud']==0]
        fraud = df[df['is_fraud']==1]

        X,y = get_X_y(fraud)
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X, y, test_size = fraud_test_size, random_state = 0)

        X,y = get_X_y(non_fraud)
        X_train_non_fraud, X_test_non_fraud, y_train_non_fraud, y_test_non_fraud = train_test_split(X, y, test_size = non_fraud_test_size, random_state = 0)

        print(f'Training set balance= [Fraud:{len(X_train_fraud)}, Non_Fraud:{len(X_train_non_fraud)}]')
        print(f'Testing set balance= [Fraud:{len(X_test_fraud)}, Non_Fraud:{len(X_test_non_fraud)}]')

        X_train= pd.concat([X_train_fraud,X_train_non_fraud])
        X_test= pd.concat([X_test_fraud,X_test_non_fraud])
        y_train= pd.concat([y_train_fraud,y_train_non_fraud])
        y_test= pd.concat([y_test_fraud,y_test_non_fraud])   
    else:
        X,y = get_X_y(df)
        test_size=kwargs['test_size']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    
    #scale X datasets:
    model_name = type(model).__name__
    fn = Path(f'output/{model_name}_ROC')
    
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    
    roc_auc = metrics.roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, label=model_name,linewidth=5)
    ax.set_title(f'ROC_AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1],'r--',linewidth=5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", prop={'size':20 })
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.figure.savefig(fn)
    st.pyplot(fig)

