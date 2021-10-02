import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tqdm




def label_fit_transform(column, enc_type="label"):
    if enc_type == "label":
        mfit = LabelEncoder()
    else:
        mfit = MinMaxScaler()
    mfit.fit(column)

    return mfit, mfit.transform(column)


def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    d = pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1])).astype(
        int)
    return pd.DataFrame(d)

def amountEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
    return pd.DataFrame(amt)


def fraudEncoder(X):
    fraud = (X == 'Yes').astype(int)
    return pd.DataFrame(fraud)

def nanNone(X):
    return X.where(pd.notnull(X), 'None')


def nanZero(X):
    return X.where(pd.notnull(X), 0)

def _quantization_binning(data, num_bins=10):
    qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
    bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
    bin_widths = np.diff(bin_edges, axis=0)
    bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
    return bin_edges, bin_centers, bin_widths

def _quantize(inputs, bin_edges, num_bins=10):
    quant_inputs = np.zeros(inputs.shape[0])
    for i, x in enumerate(inputs):
        quant_inputs[i] = np.digitize(x, bin_edges)
    quant_inputs = quant_inputs.clip(1, num_bins) - 1  # Clip edges
    return quant_inputs


def encode_data(data):

    encoded_data = data.copy(deep=True)
    
    encoder_fit = {}

    #log.info("nan resolution.")
    encoded_data['Errors?'] = nanNone(encoded_data['Errors?'])
    encoded_data['Is Fraud?'] = fraudEncoder(encoded_data['Is Fraud?'])
    encoded_data['Zip'] = nanZero(encoded_data['Zip'])
    encoded_data['Merchant State'] = nanNone(encoded_data['Merchant State'])
    encoded_data['Use Chip'] = nanNone(encoded_data['Use Chip'])
    encoded_data['Amount'] = amountEncoder(encoded_data['Amount'])

    sub_columns = ['Errors?', 'MCC', 'Zip', 'Merchant State', 'Merchant City', 'Merchant Name', 'Use Chip']

    #log.info("label-fit-transform.")
    for col_name in tqdm.tqdm(sub_columns):
        col_data = encoded_data[col_name]
        col_fit, col_data = label_fit_transform(col_data)
        encoder_fit[col_name] = col_fit
        encoded_data[col_name] = col_data
    
    #log.info("timestamp fit transform")
    timestamp = timeEncoder(encoded_data[['Year', 'Month', 'Day', 'Time']])
    timestamp_fit, timestamp = label_fit_transform(timestamp, enc_type="time")
    encoder_fit['Timestamp'] = timestamp_fit
    encoded_data['Timestamp'] = timestamp

    #log.info("timestamp quant transform")
    coldata = np.array(encoded_data['Timestamp'])
    bin_edges, bin_centers, bin_widths = _quantization_binning(coldata)
    encoded_data['Timestamp'] = _quantize(coldata, bin_edges)
    encoder_fit["Timestamp-Quant"] = [bin_edges, bin_centers, bin_widths]

    #log.info("amount quant transform")
    coldata = np.array(encoded_data['Amount'])
    bin_edges, bin_centers, bin_widths = _quantization_binning(coldata)
    encoded_data['Amount'] = _quantize(coldata, bin_edges)
    encoder_fit["Amount-Quant"] = [bin_edges, bin_centers, bin_widths]

    columns_to_select = ['User',
                            'Card',
                            'Timestamp',
                            'Amount',
                            'Use Chip',
                            'Merchant Name',
                            'Merchant City',
                            'Merchant State',
                            'Zip',
                            'MCC',
                            'Errors?',
                            'Is Fraud?']

    trans_table = encoded_data[columns_to_select]

    return encoded_data, encoder_fit, trans_table