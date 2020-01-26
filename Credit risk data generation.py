# # DATA MERGING AND VALIDATION FOR CREDIT RISK ANALYSIS

# We will import and join the datasets and explore some missingness of the data
import pandas as pd
import pickle



trans = pd.read_csv('trans.asc',sep=';')
client = pd.read_csv('client.asc',sep=';')
account = pd.read_csv('account.asc',sep=';')
disp = pd.read_csv('disp.asc',sep=';')
order = pd.read_csv('order.asc',sep=';')
loan = pd.read_csv('loan.asc',sep=';')
card = pd.read_csv('card.asc',sep=';')
district = pd.read_csv('district.asc',sep=';')


# ### Reading and merging the loan and account -related datasets
df = pd.merge(loan, account,on='account_id', suffixes=['_loan','_acnt'], how='outer')
df = pd.merge(df, disp, on='account_id', how='outer')
df = pd.merge(df, client, on='client_id', how='outer', suffixes = ['_clnt','_acnt'])
df = pd.merge(df, district, left_on='district_id_clnt', right_on='A1', how='outer')
df = pd.merge(df, card, on='disp_id', how='outer', suffixes=['', '_card'])


# ### Read in and merge transaction data
trans = pd.read_csv('trans.asc', sep=';')
trans = pd.merge(trans, df, on='account_id', suffixes=['_trans', ''], how='left')
trans = trans[~pd.isna(trans.loan_id)]

with open('transactions_data','wb') as file:
    pickle.dump(trans_loans, file)

