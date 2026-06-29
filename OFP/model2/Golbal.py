# 'timestamp', 'temperature', 'current', 'currentTXPower', 'currentRXPower', 'currentMultiRXPower1',
#          'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4', 'currentMultiTXPower1',
#          'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4', 'anomaly'
# 默认值
NaDefault = -999  # NA, 该值设置有一定问题，因为skew过小才是异常，会导致直接误判。TODO
NoneLabelTsDefault = 0  # 无异常数据的ts
TempOutlier = -255
# 时间转换
SecInHour = 3600
# 原始数据列
TimeStamp = 'timestamp'
Temperature = 'temperature'
Current = 'current'
TxPower = 'currentTXPower'
RxPower = 'currentRXPower'
RxPower1 = 'currentMultiRXPower1'
RxPower2 = 'currentMultiRXPower2'
RxPower3 = 'currentMultiRXPower3'
RxPower4 = 'currentMultiRXPower4'
TxPower1 = 'currentMultiTXPower1'
TxPower2 = 'currentMultiTXPower2'
TxPower3 = 'currentMultiTXPower3'
TxPower4 = 'currentMultiTXPower4'
Anomaly = 'anomaly'
Predict = 'predict'
Proba = 'proba'
RowIndex = 'row'
# 特征转换映射
ColumnNameMap = {
    TimeStamp: 'Ts',
    Temperature: 'Temp',
    Current: 'Curr',
    TxPower: 'TxP0',
    RxPower: 'RxP0',
    RxPower1: 'RxP1',
    RxPower2: 'RxP2',
    RxPower3: 'RxP3',
    RxPower4: 'RxP4',
    TxPower1: 'TxP1',
    TxPower2: 'TxP2',
    TxPower3: 'TxP3',
    TxPower4: 'TxP4',
    Anomaly: 'Ano',
}
NTimeStampDelta = 'TsDelta'
# 转换后特征名称
NTimeStamp = 'Ts'
NTemperature = 'Temp'
NCurrent = 'Curr'
NTxPower = 'TxP0'
NRxPower = 'RxP0'
NRxPower1 = 'RxP1'
NRxPower2 = 'RxP2'
NRxPower3 = 'RxP3'
NRxPower4 = 'RxP4'
NTxPower1 = 'TxP1'
NTxPower2 = 'TxP2'
NTxPower3 = 'TxP3'
NTxPower4 = 'TxP4'
NAnomaly = 'Ano'  # 每个时刻的标签

Label = 'Label'  # 整个光模块的标签
# 训练相关
TrainLabel = NAnomaly  # Label, NAnomaly
# 特征提取类型
MeMin = 'Min'
MeMax = 'Max'
MeDiff = 'Diff'
MeStd = 'Std'
MeSkew = 'Skew'
MeKurt = 'Kurt'
MethodHub = {MeMin, MeMax, MeDiff, MeStd, MeSkew, MeKurt}
FEATURE_FILE_PREFIX = 'feat_'
# 交叉验证
FILE_NAME = 'file_name'  # 文件名
FOLDER_INDEX = 'folder_index'  # 折号的索引，第几折
# 专家规则
RuleMatchCnt = 'RuleMatchCnt'

RuTempMin = 'RuTempMin'
RuCurrMin = 'RuCurrMin'
RuTempDiff = 'RuTempDiff'
RuCurrDiff = 'RuCurrDiff'
RuTempStd = 'RuTempStd'
RuCurrStd = 'RuCurrStd'
RuTempSkew = 'RuTempSkew'
RuTempKurt = 'RuTempKurt'
RuTxP0Min = 'RuTxP0Min'
RuRxP0Min = 'RuRxP0Min'
RuTxP0Diff = 'RuTxP0Diff'
RuRxP0Diff = 'RuRxP0Diff'
RuTxP0Std = 'RuTxP0Std'
RuRxP0Std = 'RuRxP0Std'
RuTxRxP0Skew = 'RuTxRxP0Skew'
RuTxRxP0Kurt = 'RuTxRxP0Kurt'
RuRxP1Min = 'RuRxP1Min'
RuRxP2Min = 'RuRxP2Min'
RuRxP1Diff = 'RuRxP1Diff'
RuRxP2Diff = 'RuRxP2Diff'
RuRxP1Std = 'RuRxP1Std'
RuRxP2Std = 'RuRxP2Std'
RuRxP1P2Skew = 'RuRxP1P2Skew'
RuRxP3Min = 'RuRxP3Min'
RuRxP4Min = 'RuRxP4Min'
RuRxP3Diff = 'RuRxP3Diff'
RuRxP4Diff = 'RuRxP4Diff'
RuRxP3Std = 'RuRxP3Std'
RuRxP4Std = 'RuRxP4Std'
RuRxP3P4Skew = 'RuRxP3P4Skew'
RuTxP1Min = 'RuTxP1Min'
RuTxP2Min = 'RuTxP2Min'
RuTxP1Diff = 'RuTxP1Diff'
RuTxP2Diff = 'RuTxP2Diff'
RuTxP1Std = 'RuTxP1Std'
RuTxP2Std = 'RuTxP2Std'
RuTxP3Min = 'RuTxP3Min'
RuTxP4Min = 'RuTxP4Min'
RuTxP3Diff = 'RuTxP3Diff'
RuTxP4Diff = 'RuTxP4Diff'
RuTxP3Std = 'RuTxP3Std'
RuTxP4Std = 'RuTxP4Std'
# temp<0|current<5000
# temp_diff>100|current_diff>6000
# temp_std>10|current_std>1500
# temp_skew<-20|temp_kurt>500
# TX<0|RX<0
# TX|RX_diff>1000
# TX|RX_std>500
# TX&RX_skew<-10
# TX&RX_kurt>500
# RX1|RX2<0
# RX1|RX2_diff>1000,RX1|RX2_std>200
# RX1&RX2_skew<-20
# RX3|RX4<0
# TX3|RX4_diff>1000,RX3|RX4_std>200
# RX3&RX4_skew<-20
# TX1|TX2<0
# TX1|TX2_diff>1000,TX1|TX2_std>100
# TX3|TX4<0
# TX3|TX4_diff>1000,TX3|TX4_std>100

# 特征列
FeTempMin = 'FeTempMin'
FeTempDiff = 'FeTempDiff'
FeTempStd = 'FeTempStd'
FeTempSkew = 'FeTempSkew'
FeTempKurt = 'FeTempKurt'
FeCurrentMin = 'FeCurrMin'
FeCurrentDiff = 'FeCurrDiff'
FeCurrentStd = 'FeCurrStd'
FeCurrentSkew = 'FeCurrSkew'
FeCurrentKurt = 'FeCurrKurt'
FeTxP0Min = 'FeTxP0Min'
FeTxP0Diff = 'FeTxP0Diff'
FeTxP0Std = 'FeTxP0Std'
FeTxP0Skew = 'FeTxP0Skew'
FeTxP0Kurt = 'FeTxP0Kurt'
FeRxP0Min = 'FeRxP0Min'
FeRxP0Diff = 'FeRxP0Diff'
FeRxP0Std = 'FeRxP0Std'
FeRxP0Skew = 'FeRxP0Skew'
FeRxP0Kurt = 'FeRxP0Kurt'
FeRxP1Min = 'FeRxP1Min'
FeRxP1Diff = 'FeRxP1Diff'
FeRxP1Std = 'FeRxP1Std'
FeRxP1Skew = 'FeRxP1Skew'
FeRxP1Kurt = 'FeRxP1Kurt'
FeRxP2Min = 'FeRxP2Min'
FeRxP2Diff = 'FeRxP2Diff'
FeRxP2Std = 'FeRxP2Std'
FeRxP2Skew = 'FeRxP2Skew'
FeRxP2Kurt = 'FeRxP2Kurt'
FeRxP3Min = 'FeRxP3Min'
FeRxP3Diff = 'FeRxP3Diff'
FeRxP3Std = 'FeRxP3Std'
FeRxP3Skew = 'FeRxP3Skew'
FeRxP3Kurt = 'FeRxP3Kurt'
FeRxP4Min = 'FeRxP4Min'
FeRxP4Diff = 'FeRxP4Diff'
FeRxP4Std = 'FeRxP4Std'
FeRxP4Skew = 'FeRxP4Skew'
FeRxP4Kurt = 'FeRxP4Kurt'
FeTxP1Min = 'FeTxP1Min'
FeTxP1Diff = 'FeTxP1Diff'
FeTxP1Std = 'FeTxP1Std'
FeTxP1Skew = 'FeTxP1Skew'
FeTxP1Kurt = 'FeTxP1Kurt'
FeTxP2Min = 'FeTxP2Min'
FeTxP2Diff = 'FeTxP2Diff'
FeTxP2Std = 'FeTxP2Std'
FeTxP2Skew = 'FeTxP2Skew'
FeTxP2Kurt = 'FeTxP2Kurt'
FeTxP3Min = 'FeTxP3Min'
FeTxP3Diff = 'FeTxP3Diff'
FeTxP3Std = 'FeTxP3Std'
FeTxP3Skew = 'FeTxP3Skew'
FeTxP3Kurt = 'FeTxP3Kurt'
FeTxP4Min = 'FeTxP4Min'
FeTxP4Diff = 'FeTxP4Diff'
FeTxP4Std = 'FeTxP4Std'
FeTxP4Skew = 'FeTxP4Skew'
FeTxP4Kurt = 'FeTxP4Kurt'
# 模型名称
MnBaseModel = 'MnBaseModel'
MnExtraModel = 'MnExtraModel'
MnRuleModel = 'MnRuleModel'
MnTreeModel = 'MnTreeModel'
MnMutantModel = 'MnMutantModel'

# temp<0|current<5000
# temp_diff>100|current_diff>6000
# temp_std>10|current_std>1500
# temp_std>10|current_std>1500
# TX<0|RX<0
# TX|RX_Range>1000
# TX|RX_std>500
# TX&RX_skew<-10
# TX&RX_skew>20
# RX1|RX2<0
# RX1|RX2_range>1000,RX1|RX2_std>200
# RX1&RX2<-20
# RX3|RX4<0
# TX3|RX4_range>1000,RX3|RX4_std>200
# TX1|TX2<0
# TX1|TX2_range>100,TX1|TX2_std>100
# TX3|TX4<0
# TX3|TX4_range>100,TX3|TX4_std>100



