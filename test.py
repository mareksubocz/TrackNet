#manage imports
import requests
import csv
import time
import datetime

## check bitcoin prices
def check_bitcoin_prices():
    # get bitcoin prices
    url = 'https://api.coindesk.com/v1/bpi/currentprice.json'
    response = requests.get(url)
    data = response.json()
    # get current price
    current_price = data['bpi']['USD']['rate']
    # get current time
    # print current price and time
    print('Current price: ', current_price)
    # save current price and time to csv file
    # sleep for 5 minutes
    time.sleep(300)
    
    
if __name__ == '__main__':
    while True:
        check_bitcoin_prices()