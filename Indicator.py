# Add and merge the preferred indicator into dataframe using talib 
import pandas as pd
from talib import abstract

class indicator():
    def __init__(self, stock_df):
        self.df = stock_df
        self.prices_column_name = stock_df.columns.tolist()
        self.start_time = stock_df.index[0]
        self.end_time = stock_df.index[-1]       
        pass
    
    def add_indicator(self, indicator_list:list, setting= None):
        """ Add your perfered inidicator to the dataframe
        
        Parameter:
        indicator_list : list
            The list including all the indicator you would like to add:
            ['MACD','RSI']
            
        indicator_setting : dict
            If you would like to modify default setting of the indicator,
            you can specify preferred setting and parameter into the dictionary.
            
        """
        df = self.df
        # df = df.astype('float')
        for x in indicator_list:
            if setting == None:
                output = eval('abstract.'+x+'(df)')
            else:
                try:
                    output = eval('abstract.'+x+'(df,' + setting[str(x)]+')')

                except:
                    output = eval('abstract.'+x+'(df)')
                    
            output.name = x.lower() if type(output) == pd.core.series.Series else None
            df = pd.merge(df, pd.DataFrame(output), left_on = df.index, right_on = output.index)
            # df = df.set_index('keys')

        return df