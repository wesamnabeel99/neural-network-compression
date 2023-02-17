import datetime


def get_time_format():
    now = datetime.datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")
