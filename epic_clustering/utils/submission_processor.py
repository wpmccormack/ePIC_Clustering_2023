import xxhash 
import pandas as pd

def encode_pid_column(df:pd.DataFrame, colname:str): 

        assert 'event' in df, "make sure there is a column named event, since we encode by event number" 
            
        # encode the labels to make sure it's unique across all events 
        str_ids = df['event'].astype('str') + "_" + df[colname].astype('str')
        df['clusterID'] = [xxhash.xxh64_intdigest(x, seed=0) for x in str_ids.values] 

        return df 
