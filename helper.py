
from config import Constants
import time, datetime

class Helper:
    def printline(text):
        _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
        print(f"{_date_time}   {text}")
        
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    def countries_query_builder():
        if len(Constants.selected_countries) == 0:
            return ""
        
        text = []
        sep = " or "
        for country in Constants.selected_countries:
            text.append(f'Country == "{country}"')
        query = sep.join(text)
        return query