import csv
from enum import Enum
import json
from dataclasses import dataclass
from typing import List, Optional, Generator, Dict, Tuple, Union
import copy
import time

# 行情信息
@dataclass
class MarketData:
    trading_day: str  # 交易日
    update_sec: str  # 更新时间（秒）
    update_msec: int  # 更新时间（毫秒）
    instrument_id: List[str]  # 合约ID
    volume: List[int]  # 成交量
    turn_over: List[float]  # 成交额
    open_interest: List[int]  # 持仓量
    last_price: List[float]  # 最新价
    bid_p1: List[float]  # 买一价
    bid_v1: List[int]  # 买一量
    ask_p1: List[float]  # 卖一价
    ask_v1: List[int]  # 卖一量

# 合约类型
class InstrumentType(Enum):
    FUTURE = 0
    OPTION = 1

# 期权类型
class OptionType(Enum):
    CALL = 0
    PUT = 1

# 合约静态信息
@dataclass
class InstrumentStaticData:
    trading_day: str  # 交易日
    instrument_id: str  # 合约ID
    instrument_type: InstrumentType # 合约类型 期货/期权
    tick_size: float  # 最小变动价位
    pre_settlement_price: float  # 昨结算价
    pre_close_price: float  # 昨收盘价
    upper_limit_price: float  # 涨停价
    lower_limit_price: float  # 跌停价
    expire_day: str  # 到期日
    option_type: Union[OptionType, None] # 期权类型 CALL PUT
    strike_price: Union[float, None] # 执行价

@dataclass
class SamplePrediction:
    instrument_id: List[str]  # 合约ID
    target: List[float]  # 预测目标
    update_sec: str  # 更新时间（秒）
    update_msec: int  # 更新时间（毫秒）


class CfeFinMathApi:
    def __init__(self):
        self.config_path: str = "../config.json"
        self.option_month: str = "IO2507"
        self.date: str = ""
        self.data_path: str = ""
        self.future_md: List[MarketData] = []
        self.future_static_md: List[InstrumentStaticData]
        self.option_md: List[MarketData] = []
        self.option_static_md: List[InstrumentStaticData]
        self.status: str = 'initialized'
        self.predictions: List[SamplePrediction] = []
        self.opt_ins: List[str] = []
        self.__init_api()
        self.__read_csv()

    def get_option_static_md(self) -> List[InstrumentStaticData]:
        return self.option_static_md

    def get_future_static_md(self) -> List[InstrumentStaticData]:
        return self.future_static_md

    def iter_test(self) -> Generator[Optional[Tuple[MarketData, Optional[MarketData], SamplePrediction]], None, None]:
        if self.status != 'initialized':
            raise Exception(
                'WARNING: the real API can only iterate over `iter_test()` once.')
        start_time = time.time()

        for i in range(len(self.future_md)):
            cur_future_md: MarketData = self.future_md[i]
            # check one minute iter option md
            minute: str = cur_future_md.update_sec
            cur_min: int = 3600 * \
                int(minute[0:2]) + 60 * int(minute[3:5]) + int(minute[6:8])

            cur_option_md: Optional[MarketData] = None

            if (cur_min % 600 == 0 and cur_future_md.update_msec == 0) or (cur_min == 3600 * 9 + 29 * 60):
                cur_option_md = self.option_md[i]

            self.status = 'prediction_needed'
            sample_prediction: SamplePrediction = SamplePrediction(
                instrument_id=self.opt_ins,
                target=[0] * len(self.opt_ins),
                update_sec=cur_future_md.update_sec,
                update_msec=cur_future_md.update_msec
            )
            yield (cur_future_md, cur_option_md, sample_prediction)
            while self.status != 'prediction_received':
                print(
                    'You must call `predict()` successfully before you can continue with `iter_test()`', flush=True)
                yield None
                
        end_time = time.time()
        running_time = end_time - start_time

        # 保存时间统计到文件
        timing_result = {
            "iter_test_running_time": running_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        with open("iter_test_timing.json", "w") as f:
            json.dump(timing_result, f, indent=2)

        self.__write_submission_csv()
        self.status = 'finished'

    def predict(self, user_predictions: SamplePrediction) -> None:
        if self.status == 'finished':
            raise Exception(
                'You have already made predictions for the full test set.')
        if self.status != 'prediction_needed':
            raise Exception(
                'You must get the next test sample from `iter_test()` first.')
        user_predictions.target = copy.deepcopy(user_predictions.target)
        self.predictions.append(user_predictions)
        self.status = 'prediction_received'

    def __init_api(self) -> None:
        with open(self.config_path) as f:
            config: str = f.read()
            config_json: Dict[str, str] = json.loads(config)
            self.date = config_json["date"]
            self.data_path = config_json["data_path"]

    def __read_csv(self) -> None:
        future_md_path: str = f"{self.data_path}/IF_{self.date}.csv"
        future_md_static_path: str = f"{self.data_path}/IF_{self.date}_static.csv"
        option_md_path: str = f"{self.data_path}/{self.option_month}_{self.date}.csv"
        option_md_static_path: str = f"{self.data_path}/{self.option_month}_{self.date}_static.csv"

        self.future_static_md = self.__read_static_data_csv(future_md_static_path, InstrumentType.FUTURE)
        self.option_static_md = self.__read_static_data_csv(option_md_static_path, InstrumentType.OPTION)

        self.__read_market_data_csv(future_md_path, len(self.future_static_md), self.future_md)
        self.__read_market_data_csv(option_md_path, len(self.option_static_md), self.option_md)

        self.opt_ins = [data.instrument_id for data in self.option_static_md]

    def __read_market_data_csv(self, file_path: str, group_cnt: int, all_datas: List[MarketData]) -> None:
        with open(file_path, mode='r') as csv_file:
            csv_reader: csv.reader = csv.reader(csv_file)
            header: List[str] = next(csv_reader) 

            current_data: Optional[MarketData] = None
            for i, row in enumerate(csv_reader):
                if (i % group_cnt) == 0:
                    if current_data:
                        all_datas.append(current_data)

                    current_data = MarketData(
                        trading_day=row[0], update_sec=row[1], update_msec=int(row[2]),
                        instrument_id=[], volume=[], turn_over=[], open_interest=[],
                        last_price=[], bid_p1=[], bid_v1=[], ask_p1=[], ask_v1=[]
                    )

                current_data.instrument_id.append(row[3])
                current_data.volume.append(int(row[4]))
                current_data.turn_over.append(float(row[5]))
                current_data.open_interest.append(int(row[6]))
                current_data.last_price.append(float(row[7]))
                current_data.bid_p1.append(float(row[8]))
                current_data.bid_v1.append(int(row[9]))
                current_data.ask_p1.append(float(row[10]))
                current_data.ask_v1.append(int(row[11]))

            if current_data:
                all_datas.append(current_data)

    def __read_static_data_csv(self, file_path: str, ins_type: InstrumentType) -> List[InstrumentStaticData]:
        data: List[InstrumentStaticData] = []
        with open(file_path, mode='r') as csv_file:
            csv_reader: csv.DictReader = csv.DictReader(csv_file)
            for row in csv_reader:
                if "option_type" not in row.keys():
                    option_type = None
                    strike_price = None
                else:
                    option_type = OptionType.CALL if row["option_type"] == "C" else OptionType.PUT
                    strike_price = float(row["strike_price"])
                data.append(InstrumentStaticData(
                    trading_day=row["trading_day"],
                    instrument_id=row['instrument_id'],
                    tick_size=float(row["tick_size"]),
                    pre_settlement_price=float(row['pre_settlement_price']),
                    pre_close_price=float(row['pre_close_price']),
                    upper_limit_price=float(row['upper_limit_price']),
                    lower_limit_price=float(row['lower_limit_price']),
                    expire_day=row['expire_day'],
                    instrument_type=ins_type,
                    option_type = option_type,
                    strike_price = strike_price
                ))
        return data
    
    def __write_submission_csv(self) -> None:
        with open('submission.csv', 'w', newline='') as f_open:
            writer: csv.DictWriter = csv.DictWriter(
                f_open, fieldnames=['instrument', 'target', 'update_sec', 'update_msec'])
            writer.writeheader()

            for prediction in self.predictions:
                for instrument, target in zip(prediction.instrument_id, prediction.target):
                    writer.writerow({
                        'instrument': instrument,
                        'target': target,
                        'update_sec': prediction.update_sec,
                        'update_msec': prediction.update_msec
                    })
