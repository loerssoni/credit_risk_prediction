
# # DATA MERGING AND VALIDATION FOR CREDIT RISK ANALYSIS

# We will begin by importing relevant libraries, importing and joining the datasets and exploring some missingness of the data


import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import numpy as np
import sys



trans = pd.read_csv('trans.asc',sep=';')
client = pd.read_csv('client.asc',sep=';')
account = pd.read_csv('account.asc',sep=';')
disp = pd.read_csv('disp.asc',sep=';')
order = pd.read_csv('order.asc',sep=';')
loan = pd.read_csv('loan.asc',sep=';')
card = pd.read_csv('card.asc',sep=';')
district = pd.read_csv('district.asc',sep=';')

# ### Reading and merging the loan and account -related datasets
# We merge the static datasets and explore proportional missingness in the full data.


df = pd.merge(loan, account,on='account_id', suffixes=['_loan','_acnt'], how='outer')
df = pd.merge(df, disp, on='account_id', how='outer')
df = pd.merge(df, client, on='client_id', how='outer', suffixes = ['_clnt','_acnt'])
df = pd.merge(df, district, left_on='district_id_clnt', right_on='A1', how='outer')
df = pd.merge(df, card, on='disp_id', how='outer', suffixes=['', '_card'])

# ### Feature generation for loans
# 
# We first drop observations on users that do not have a loan associated in any of the accounts they participate in.
loans = df[~pd.isna(df.loan_id)]
# We also drop some irrelevant columns. Most of the identification was only necessary for joining the data, so they are dropped. Most of the demographic data describes essentially the population and urbanization in the area, so this redundant information is dropped.
# There are only 5 junior and 3 gold cards in the data so the card type is dropped as well.
loans.drop(['account_id','district_id_acnt', 'A1', 'A2', 'A3',
            'A5','A6', 'A7', 'A8', 'A9','district_id_clnt', 'disp_id',
            'client_id', 'card_id', 'type_card'], axis=1, inplace=True)

# We then encode features into formats suitable for machine learning.

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


loans.columns = ['amount', 'duration', 'payments', 'pop', 'urban_rat', 'avg_sal', 'rat_urban', 'card',
       'target', 'gender', 'appl_age', 'accnt_age',
       'freq_trans', 'freq_weekly', 'unempl',
       'crime', 'multi']


# ### Read in and merge transaction data
# 
# Note that, because the dataset is relatively small, we introduce some redundancy for a while by joining the full dataset of static information. 
# This is done to spare lines of code.

trans = pd.read_csv('trans.asc', sep=';')

trans = pd.merge(trans, df, on='account_id', suffixes=['_trans', ''], how='left')
#Subset the data set to transactions for accounts with loans:
trans_loans = trans[~pd.isna(trans.loan_id)]


# In order to work with the dates, we will first transform them to datetime format

trans_loans.date = pd.to_datetime(trans_loans.date, format='%y%m%d')
trans_loans.date_loan = pd.to_datetime(trans_loans.date_loan, format='%y%m%d')

#filter to transactions prior to loan issuance
trans_loans = trans_loans[trans_loans.date < trans_loans.date_loan]

# ## Aggregating transactions
# 
# For now, we'll drop the other info and focus on aggregating the transactions data to loan-level.
# 
# First, we drop irrelevant columns and generate some dummy variables.

trans_loans = trans_loans[['loan_id', 'date','date_loan', 'type_trans', 'operation',
       'amount_trans', 'balance', 'k_symbol']]

trans_loans = trans_loans[trans_loans.k_symbol != 'UROK']
trans_loans = trans_loans[trans_loans.k_symbol != 'SLUZBY']
trans_loans.k_symbol.fillna(trans_loans.operation, inplace=True)

trans_loans['b_deposit'] = trans_loans.k_symbol.str.contains('PREVOD Z').astype(int)
trans_loans['c_deposit'] = trans_loans.k_symbol.str.contains('VKLAD').astype(int)
trans_loans['withdr'] = trans_loans.k_symbol.str.contains('VYBER').astype(int)
trans_loans['sanc'] = trans_loans.k_symbol.str.contains('SANK').astype(int)
trans_loans['b_withdr'] = trans_loans.operation.str.contains('PREVOD NA').astype(int)

trans_loans.drop(['type_trans','operation','k_symbol'], axis=1, inplace=True)



def aggregate(data, time_window_max = 100, time_window_min = 0):  
    trans_agg = data[(data.date_loan - data.date).dt.days < time_window_max]
    trans_agg = trans_agg[(data.date_loan - data.date).dt.days >= time_window_min]
    trans_agg['balance_start'] = trans_agg.balance - trans_agg.amount_trans
    trans_agg['transactions'] = 1
    trans_agg['net_cdeposit'] = trans_agg.c_deposit * trans_agg.amount_trans - trans_agg.withdr * trans_agg.amount_trans
    trans_agg['net_bdeposit'] =  trans_agg.b_deposit * trans_agg.amount_trans - trans_agg.b_withdr * trans_agg.amount_trans

    trans_agg = trans_agg.groupby('loan_id').agg({
        'balance':['min','mean','max'],
        'c_deposit':'sum',
        'withdr':'sum',
        'sanc':'max',
        'balance_start':lambda x: x.iloc[0],
        'transactions':'sum',
        'net_cdeposit':'sum',
        'net_bdeposit':['sum','max']
    })
    trans_agg.columns = ['_'.join(col).strip() for col in trans_agg.columns.values]

    trans_agg['balance_max'] = trans_agg[['balance_max', 'balance_start_<lambda>']].max(axis=1)
    trans_agg['balance_min'] = trans_agg[['balance_min', 'balance_start_<lambda>']].min(axis=1)
    trans_agg.drop(['balance_start_<lambda>'], axis=1, inplace=True)
    
    return trans_agg


if __name__== "__main__":
    max_time = 3000
    min_time = 60
    if len(sys.argv) == 3:
        max_time = int(sys.argv[1])
        min_time = int(sys.argv[2])
        
    trans_agg = aggregate(trans_loans, max_time, min_time)
    trans_agg2 = aggregate(trans_loans, min_time, 0)
    trans_agg = trans_agg.join(trans_agg2, rsuffix='_short')
    final = pd.merge(loans, trans_agg, left_index=True, right_index=True, suffixes=['','_trans'], how='left')

    with open('loan_data','wb') as file:
        pickle.dump(final, file)



