#!/usr/bin/env python
# coding: utf-8

# # DATA MERGING AND VALIDATION FOR CREDIT RISK ANALYSIS

# We will begin by importing relevant libraries, importing and joining the datasets and exploring some missingness of the data
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import numpy as np




trans = pd.read_csv('trans.asc',sep=';')
client = pd.read_csv('client.asc',sep=';')
account = pd.read_csv('account.asc',sep=';')
disp = pd.read_csv('disp.asc',sep=';')
order = pd.read_csv('order.asc',sep=';')
loan = pd.read_csv('loan.asc',sep=';')
card = pd.read_csv('card.asc',sep=';')
district = pd.read_csv('district.asc',sep=';')


# ### Reading and merging the loan and account -related datasets
# We merge the static datasets
df = pd.merge(loan, account,on='account_id', suffixes=['_loan','_acnt'], how='outer')
df = pd.merge(df, disp, on='account_id', how='outer')
df = pd.merge(df, client, on='client_id', how='outer', suffixes = ['_clnt','_acnt'])
df = pd.merge(df, district, left_on='district_id_clnt', right_on='A1', how='outer')
df = pd.merge(df, card, on='disp_id', how='outer', suffixes=['', '_card'])

# ### Feature generation for loans
# 
#drop non-loan observations
loans = df[~pd.isna(df.loan_id)]

#drop irrelevant columns
loans.drop(['account_id','district_id_acnt', 'A1', 'A2', 'A3',
            'A5','A6', 'A7', 'A8', 'A9','district_id_clnt', 'disp_id',
            'client_id', 'card_id', 'type_card'], axis=1, inplace=True)


#create dummy for whether the loan completed successfully
loans['target'] = (loans.status == 'B').astype(int) + (loans.status == 'D').astype(int)

#convert date columns to datetime
loans['date_loan'] = pd.to_datetime(loans.date_loan, format='%y%m%d')
loans['issued'] = pd.to_datetime(loans.issued.str[:6])
loans['date_acnt'] = pd.to_datetime(loans.date_acnt, format='%y%m%d')

#find gender (encoded into the birthnumber) and convert birthdate into datetime
loans['gender'] = (loans.birth_number % 10000 > 5000).astype(int)
loans['birthdate'] = loans.birth_number - 5000 * loans.gender + 19000000
loans['birthdate'] = pd.to_datetime(loans.birthdate, format='%Y%m%d')

#find the age of applicant and the account at the time of loan issuance
loans['appl_age'] = (loans.date_loan - loans.birthdate).dt.days / 365.25
loans['accnt_age'] = (loans.date_loan - loans.date_acnt).dt.days / 365.25



#create dummy for whether the account has an associated card at the time of loan issuance
loans['issued'] = (loans.issued < loans.date_loan).astype(int)

#create dummies for the frequency of statement issuance and the account type
loans = pd.get_dummies(loans, columns=['frequency', 'type'], drop_first=True)


# select unemployment and crime from the demographic statistics.
loans['unempl'] = np.select([loans.date_loan.dt.year > 1996,
                             loans.date_loan.dt.year < 1997],
                            [loans.A13, loans.A12])

loans['crime'] = np.select([loans.date_loan.dt.year > 1996, loans.date_loan.dt.year < 1997],
          [loans.A16, loans.A15])

# convert the columns to numeric values and scale the crime numbers for population
loans['unempl'] = pd.to_numeric(loans.unempl, errors='coerce')
loans['crime'] = pd.to_numeric(loans.crime, errors='coerce') / loans.A4
loans['A14'] = loans.A14 / loans.A4 * 100

#finally, aggregate to loan-level from client-level data
loans = loans.groupby('loan_id').agg('mean')
loans['multi'] = np.select([loans.type_OWNER < 1], [1], 0)

#drop unnecessary columns
loans.drop(['birth_number','type_OWNER','A13','A16'], axis=1, inplace=True)


# In[38]:


loans.columns = ['amount', 'duration', 'payments', 'pop', 'urban_rat', 'avg_sal', 'rat_urban', 'card',
       'target', 'gender', 'appl_age', 'accnt_age',
       'freq_trans', 'freq_weekly', 'unempl',
       'crime', 'multi']


# ### Read in and merge transaction data
# 
# Note that, because the dataset is relatively small, we introduce some redundancy for a while by joining the full dataset of static information. 
trans = pd.read_csv('trans.asc', sep=';')


#merge transactions with static data
trans = pd.merge(trans, df, on='account_id', suffixes=['_trans', ''], how='left')
#Subset the data set to transactions for accounts with loans:
trans_loans = trans[~pd.isna(trans.loan_id)]


# ### Datetime wrangling
# 
# Because we're interested in predicting bad loans, we should use transaction data from only prior to giving out the loan.
# In order to work with the dates, we will first transform them to datetime forma
trans_loans.date = pd.to_datetime(trans_loans.date, format='%y%m%d')
trans_loans.date_loan = pd.to_datetime(trans_loans.date_loan, format='%y%m%d')

#filter to transactions prior to loan issuance
trans_loans = trans_loans[trans_loans.date < trans_loans.date_loan]


# ## Aggregating transactions
# 
# For now, we'll drop the other info and focus on aggregating the transactions data to loan-level.


trans_loans = trans_loans[['loan_id', 'date','date_loan', 'type_trans', 'operation',
       'amount_trans', 'balance', 'k_symbol', 'bank', 'account']]


trans_loans = pd.get_dummies(trans_loans, columns=['type_trans', 'operation','k_symbol'])
trans_loans['bank'] = trans_loans.bank.isna().astype('int')
trans_loans['account'] = trans_loans.account.isna().astype('int')


trans_agg = trans_loans.groupby('loan_id').agg('mean')

trans_agg = trans_agg.join(trans_loans.loan_id.value_counts(sort=False))



# ### Merge transaction and loan datasets, export

final = pd.merge(loans, trans_agg, left_index=True, right_index=True, suffixes=['','_trans'], how='left')
with open('loan_data','wb') as file:
    pickle.dump(final, file)





