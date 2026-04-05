# coding: utf-8
import pandas as pd


def update_industry_numbers():
    # 定义行业与编号的映射字典
    industry_mapping = {
        "农、林、牧、渔业": "01",
        "采矿业": "02",
        "制造业": "16",
        "电力、热力、燃气及水生产和供应业": "03",
        "建筑业": "05",
        "交通运输、仓储和邮政业": "06",
        "信息传输、软件和信息技术服务业": "15",
        "金融业": "08",
        "房地产业": "04",
        "租赁和商务服务业": "19",
        "综合": "18",
        "批发和零售业": "11",
        "水利、环境和公共设施管理业": "12",
        "文化、体育和娱乐业": "14",
        "居民服务、修理和其他服务业": "09",
        "卫生和社会工作": "13",
        "教育": "07",
        "住宿和餐饮业": "17",
        "科学研究和技术服务业": "10"
    }

    try:
        # 读取CSV文件，使用GBK编码
        df = pd.read_csv('AllData/Data2012.csv', encoding='gbk')

        # 检查必要的列是否存在
        if 'Industry' not in df.columns:
            print("错误：CSV文件中没有'Industry'列")
            return

        # 根据Industry列的内容填充Number列
        if 'Number' in df.columns:
            # 如果Number列已存在，则更新它
            df['Number'] = df['Industry'].map(industry_mapping)
        else:
            # 如果Number列不存在，则创建它
            df['Number'] = df['Industry'].map(industry_mapping)

        # 检查是否有未匹配的行业
        unmatched_industries = df[df['Number'].isna()]['Industry'].unique()
        if len(unmatched_industries) > 0:
            print("警告：以下行业没有找到对应的编号：")
            for industry in unmatched_industries:
                print(f"  '{industry}'")

        # 直接覆盖原文件，使用GBK编码
        df.to_csv('AllData/Data2012.csv', index=False, encoding='gbk')
        print("行业编号已成功更新并保存到Data.csv（GBK编码）")

        # 显示更新统计信息
        total_rows = len(df)
        matched_rows = df['Number'].notna().sum()
        print(f"总共处理 {total_rows} 行数据，成功匹配 {matched_rows} 行")

    except FileNotFoundError:
        print("错误：找不到Data.csv文件")
    except UnicodeDecodeError:
        print("错误：使用GBK编码读取文件失败，请检查文件编码")
    except Exception as e:
        print(f"处理过程中出现错误：{e}")


# 运行程序
if __name__ == "__main__":
    update_industry_numbers()