import math


def convert_dbm2mw(dbm):
    mw = 10**(dbm/10)
    return mw


def convert_mw2dbm(mw):
    dbm = 10*math.log10(mw)
    return dbm


def cal_total_power(p1, p2, p3, p4):
    mw1 = convert_dbm2mw(p1)
    mw2 = convert_dbm2mw(p2)
    mw3 = convert_dbm2mw(p3)
    mw4 = convert_dbm2mw(p4)
    total_mw = mw1 + mw2 + mw3 + mw4
    total_power_dbm = convert_mw2dbm(total_mw)
    return total_power_dbm


# 123.82,41.16,101.61,117.34
# 111.06,	127.04,	102.02,	107.62
# currentMultiRXPower1 (dBm)
# 多路光纤接收功率1
# 浮点数
# -1.16
# currentMultiRXPower2 (dBm)
# 多路光纤接收功率2
# 浮点数
# -1.11
# currentMultiRXPower3 (dBm)
# 多路光纤接收功率3
# 浮点数
# -4.06
# currentMultiRXPower4 (dBm)
# 多路光纤接收功率4
# 浮点数
# -0.78
# currentMultiTXPower1(dBm)
# 多路光纤发送功率1
# 浮点数
# -0.84
# currentMultiTXPower2(dBm)
# 多路光纤发送功率2
# 浮点数
# -0.41
# currentMultiTXPower3(dBm)
# 多路光纤发送功率3
# 浮点    
# -0.36
# currentMultiTXPower4(dBm) 
# 多路光纤发送功率4
# 浮点数
# 0.31

if __name__ == '__main__':
    pass


