import pandas as pd
import os




#define the ReadCSV function:
def ReadCSVfiles(files_path, wannaprint=False): #in the files_path argument, you should pass the path to the folder containing the csv files
    files = os.listdir(files_path)
    csv_files = [os.path.join(files_path, file) for file in files] #this joins the path with the file name

    def ReadFile(x):
        df = pd.read_csv(x, parse_dates=True, index_col=0)
        df = df.loc[: , ["close"]]
        return df

    # Use list comprehension to read all CSV files
    dataframes = [ReadFile(file) for file in csv_files]
    # Merge all DataFrames into a single DataFrame based on the index
    merged_df = pd.concat(dataframes, axis=1)
    merged_df.columns = files #assign column names to the merged dataframe
    merged_df.dropna(inplace=True) #drop any rows with missing values

    if wannaprint == True:
        print(merged_df.head())

    return merged_df

