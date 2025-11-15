from typing import List, Optional, Generator, Tuple

from cfe_fin_math_api import CfeFinMathApi, InstrumentStaticData, MarketData, SamplePrediction
import math
api: CfeFinMathApi = CfeFinMathApi()
opt_static: List[InstrumentStaticData] = api.get_option_static_md()
fut_static: List[InstrumentStaticData] = api.get_future_static_md()

iter_test = api.iter_test()
target_op = [0] * len(opt_static)
for future_md, option_md, predict in iter_test:
    if option_md:
        for i in range(len(option_md.bid_p1)):
            if math.isnan(option_md.bid_p1[i]):
                mid_p = option_md.ask_p1[i]
            elif math.isnan(option_md.ask_p1[i]):
                mid_p = option_md.bid_p1[i]
            else:
              mid_p = (option_md.bid_p1[i]  + option_md.ask_p1[i]) * 0.5
            target_op[i] = mid_p
    predict.target = target_op
    api.predict(predict)
