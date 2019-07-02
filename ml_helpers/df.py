from .imports import *

# Taken from fast.ai - https://github.com/fastai/fastai
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)