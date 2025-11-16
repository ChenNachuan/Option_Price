import sys
from cfe_fin_math_api import CfeFinMathApi, MarketData, SamplePrediction
from pricing_engine import PricingEngine

def main():
    """
    项目主入口
    """
    try:
        # 1. 初始化API
        api = CfeFinMathApi()
        
        # 2. 初始化定价引擎
        engine = PricingEngine(api)

        # 3. 创建数据迭代器
        iter_test = api.iter_test()

        print("--- 开始实训循环 ---")
        
        # 4. 循环处理行情
        # 将 'predict' 重命名为 'predict_obj' 以避免混淆
        for data_tuple in iter_test:
            
            # *** 新增：检查 API 是否返回 None ***
            # 如果 api.predict() 没有被调用，iter_test 会 yield None
            if data_tuple is None:
                print("API yielded None. 可能是 api.predict() 未被调用。")
                break # 退出循环

            future_md, option_md, predict_obj = data_tuple
            
            # (A) 校准阶段：
            if option_md:
                try:
                    engine.calibrate(option_md, future_md)
                except Exception as e:
                    print(f"校准时发生严重错误: {e}")

            # (B) 预测阶段：
            try:
                engine.predict(future_md, predict_obj)
            except Exception as e:
                print(f"预测时发生严重错误: {e}")
                # 确保在出错时也填充一个有效列表，避免API崩溃
                predict_obj.target = [0.0] * len(engine.opt_static_list)

            # (C) 提交预测：
            # *** 这是必需的步骤，用于通知API已完成预测 ***
            api.predict(predict_obj)


        print("--- 实训循环结束 ---")

    except FileNotFoundError:
        print("错误：无法找到数据文件。")
        print("请确保 'config.json' 配置正确，且 'data_path' 指向的路径下有数据。")
    except Exception as e:
        print(f"程序运行中发生未捕获的异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()