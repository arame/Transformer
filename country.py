from config import Hyper
from helper import Helper

class Country:
    def __init__(self, df) -> None:
        self.df = df

    def print_balance(self):
        for _country in Hyper.selected_countries:
            self.print_balance_line(_country)
            
    def print_balance_line(self, _country):
        perc = self.calc_percentage(_country)
        Helper.printline(f"{_country} = {perc} %")

    def calc_percentage(self, value):
        return round(float(len(self.df.query(f'Country == "{value}"')) / len(self.df)) * 100, 2)