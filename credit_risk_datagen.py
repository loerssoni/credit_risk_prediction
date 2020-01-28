
# # DATA MERGING AND VALIDATION FOR CREDIT RISK ANALYSIS
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import numpy as np
import sys

# ### Reading and merging the loan and account -related datasets
# We merge the static datasets and explore proportional missingness in the full data.

client = pd.read_csv('client.asc',sep=';')
account = pd.read_csv('account.asc',sep=';')
disp = pd.read_csv('disp.asc',sep=';')
order = pd.read_csv('order.asc',sep=';')
loan = pd.read_csv('loan.asc',sep=';')
card = pd.read_csv('card.asc',sep=';')
district = pd.read_csv('district.asc',sep=';')

df = pd.merge(loan, account,on='account_id', suffixes=['_loan','_acnt'], how='outer')
df = pd.merge(df, disp, on='account_id', how='outer')
df = pd.merge(df, client, on='client_id', how='outer', suffixes = ['_clnt','_acnt'])
df = pd.merge(df, district, left_on='district_id_clnt', right_on='A1', how='outer')
df = pd.merge(df, card, on='disp_id', how='outer', suffixes=['', '_card'])


# ### Feature generation for loans
# 
# We first drop observations on users that do not have a loan associated in any of the accounts they participate in.
# 
# We also drop some irrelevant columns. Most of the identification was only necessary for joining the data, so they are dropped. 
# There are only 5 junior and 3 gold cards in the data so the card type is dropped as well.
# 
# We then encode features into formats suitable for machine learning.
# 
# The demographic data is available only as static values measured after some of the loans in the data are already issued. 
# Because ex-ante values are not available, we make the assumption that  the demographics do not drastically change across years. 
# However, if the demographics turn out to be important in predicting credit defaults, this problem should be readdressed.

loans = df[~df.loan_id.isna()]

# create a table that we will later use for merging
loan_dates = loans[['account_id','loan_id','date_loan']]
#translate loan date to datetime format
loan_dates.date_loan = pd.to_datetime(loan_dates.date_loan, format='%y%m%d')

#drop unnecessary columns
loans.drop(['account_id','district_id_acnt','district_id_clnt', 'disp_id',
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
loans['A13'] = np.select([loans.date_loan.dt.year > 1996,
                             loans.date_loan.dt.year < 1997],
                            [loans.A13, loans.A12])

loans['A12'] = np.select([loans.date_loan.dt.year > 1996, loans.date_loan.dt.year < 1997],
          [loans.A16, loans.A15])

# convert the columns to numeric values and scale the crime numbers for population
loans['A13'] = pd.to_numeric(loans.A13, errors='coerce')
loans['A12'] = pd.to_numeric(loans.A12, errors='coerce') / loans.A4
loans['A14'] = loans.A14 / loans.A4 * 100

#finally, aggregate to loan-level from client-level data
loans = loans.groupby('loan_id').agg('mean')
#create dummy for loans, where the account has multiple users
loans['multi'] = np.select([loans.type_OWNER < 1], [1], 0)

#drop unnecessary columns
loans.drop(['birth_number','type_OWNER','A16'], axis=1, inplace=True)

#rename columns
loans.columns = ['amount', 'duration', 'payments', 'A1', 'A4', 'A5', 'A6', 'A7', 'A8',
       'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'issued', 'target', 'gender',
       'appl_age', 'accnt_age', 'frequency_trans',
       'frequency_weekly', 'multi']


#address missing employment and crime values
loans.loc[loans.A13.isna(),['A12','A13']] = loans.loc[5281,['A12','A13']].values
loans.drop('A1', axis=1, inplace=True)


# ### Read in and merge transaction data
trans = pd.read_csv('trans.asc', sep=';')
trans_loans = pd.merge(loan_dates, trans, on='account_id', suffixes=['', '_trans'], how='left')

# Because we're interested in predicting bad loans, we should use transaction data from only prior to giving out the loan.
# In order to work with the dates, we will first transform them to datetime format
trans_loans.date = pd.to_datetime(trans_loans.date, format='%y%m%d')

#filter to transactions prior to loan issuance
trans_loans = trans_loans[trans_loans.date < trans_loans.date_loan]


# ## Aggregating transactions
# 
# We noe move on to aggregating the transactions data to loan-level.
# 
# First, we drop irrelevant columns and generate some dummy variables. Only transaction-related variables are kept.
# Out of the transaction variables, the bank and account columns are likely not relevant and quite sparse, so they are dropped. 

trans_loans = trans_loans[['loan_id', 'date','date_loan', 'type', 'operation',
       'amount', 'balance', 'k_symbol']]

#drop transactions of payments for statements as they are the same price for all customers
#and very likely not of interest
trans_loans = trans_loans[trans_loans.k_symbol != 'SLUZBY']

#as the k_symbol and operation cover similar things, we combine them into one
#categorical variable that describes the operation type
trans_loans.k_symbol.fillna(trans_loans.operation, inplace=True)

#get dummies for operation
trans_loans = pd.get_dummies(trans_loans, columns=['k_symbol'], prefix='type')

#combine card and cash withdrawals to one
trans_loans['type_VYBER'] = trans_loans['type_VYBER KARTOU'] + trans_loans['type_VYBER']

#drop irrelevant columns and rename columns more intuitively
trans_loans.drop(['type','operation', 'type_VYBER KARTOU'], axis=1, inplace=True)
trans_loans.columns = ['loan_id', 'date', 'date_loan', 'amount_trans', 'balance', 'b_withdr',
       'insur', 'b_deposit', 'sanc', 'hhold',
       'interest', 'c_deposit', 'c_withdr']


# ### Create aggregate time-series variables
# We will create aggregate time-series' for the recent interest rate as measured by the rate transactions and for the amount of concurrent loan applicants (rolling mean of past 6 months.

#create interest-rate variable for interest transactions
trans_loans['rate'] = (trans_loans.amount_trans * trans_loans.interest) / (trans_loans.balance - trans_loans.amount_trans)
rates = trans_loans[trans_loans.interest == 1].groupby('date').mean().rate

#create applicant variable as the rolling 90-day average of applicants
applicants = trans_loans.groupby('loan_id').first().date_loan.value_counts().sort_index()
applicants = applicants.resample('D').sum().rolling(90).sum().shift(1)

#Merge time-series variables with static loan data
ts_data = loan_dates.sort_values('date_loan')
#merge rates
ts_data = pd.merge_asof(ts_data, rates, left_on='date_loan', right_index=True)
#merge amount of applicants
ts_data = pd.merge(ts_data, applicants, left_on='date_loan', right_index=True)

#drop trash and set index for joining
ts_data.drop(['date_loan','account_id','date_loan_x'], axis=1, inplace=True)
ts_data.columns=['loan_id', 'rate','applicants']
ts_data.set_index('loan_id', inplace=True)

loans = loans.join(ts_data)

# ### Compute aggregations and combine
# We first aggregate the transaction data to loan-level.
# Then we merge with the static loan data and the time-series variables.

def aggregate(data, time_window_max = 100, time_window_min = 0):  
    trans_agg = data[(data.date_loan - data.date).dt.days < time_window_max]
    trans_agg = trans_agg[(data.date_loan - data.date).dt.days > time_window_min]
    trans_agg['balance_start'] = trans_agg.balance - trans_agg.amount_trans
    trans_agg['transactions'] = 1
    trans_agg['net_cdeposit'] = trans_agg.c_deposit * trans_agg.amount_trans                                 - trans_agg.c_withdr * trans_agg.amount_trans
    trans_agg['net_bdeposit'] =  trans_agg.b_deposit * trans_agg.amount_trans                                 - trans_agg.b_withdr * trans_agg.amount_trans
    trans_agg['age'] = (trans_loans.date_loan - trans_loans.date).dt.days
    trans_agg = trans_agg.groupby('loan_id').agg({
        'balance':['min','mean','max'],
        
        'c_deposit':'sum',
        'c_withdr':'sum',
        'sanc':'max',
        'rate': 'max',
        'balance_start':lambda x: x.iloc[0],
        'transactions':'sum',
        'net_cdeposit':'sum',
        'net_bdeposit':['sum','max'],
        'age':'mean'
    })
    trans_agg.columns = ['_'.join(col).strip() for col in trans_agg.columns.values]

    trans_agg['balance_max'] = trans_agg[['balance_max', 'balance_start_<lambda>']].max(axis=1)
    trans_agg['balance_min'] = trans_agg[['balance_min', 'balance_start_<lambda>']].min(axis=1)
    trans_agg.drop(['balance_start_<lambda>'], axis=1, inplace=True)
    trans_agg['neg_bal'] = (trans_agg.balance_min < 0).astype(int)
    return trans_agg


# Finally, we create aggregations. Here, we use a full history up until the most recent two months and the recent two months separately.


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
