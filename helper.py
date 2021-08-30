
from config import Constants, Hyper
import time, datetime, os, random
import numpy as np
import torch as T

class Helper:
    def printline(text):
        _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
        print(f"{_date_time}   {text}")
    
    def printlines(text, number_of_lines):
        if number_of_lines < 2:
            Helper.printline(text)
            return
        
        new_lines = ""
        for _ in range(number_of_lines - 1):
            new_lines += "\n"
        _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
        print(f"{new_lines}{_date_time}   {text}") 
          
    def check_folder(folder):
        if os.path.isdir(folder) == False:
            os.mkdir(folder)
            
    def remove_file(file):
        if os.path.isfile(file):
            os.remove(file)
        
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    def countries_query_builder():
        if len(Hyper.selected_countries) == 0:
            return ""
        
        text = []
        sep = " or "
        for country in Hyper.selected_countries:
            text.append(f'Country == "{country}"')
        query = sep.join(text)
        return query
    
    def set_seed():
        random.seed(Constants.seed_val)
        np.random.seed(Constants.seed_val)
        T.manual_seed(Constants.seed_val)
        T.cuda.manual_seed_all(Constants.seed_val)
        
    def time_lapse(total_t0):
        return Helper.format_time(time.time()-total_t0)