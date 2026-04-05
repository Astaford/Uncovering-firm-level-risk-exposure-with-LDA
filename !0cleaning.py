# coding: utf-8
import pandas as pd
import re


def strict_gbk_conversion():
    """严格的GBK转换，确保100%兼容"""
    input_file = 'AllData/Data2015.csv'

    print("执行严格GBK转换...")

    # 读取原始文件
    df = pd.read_csv(input_file, encoding='latin-1')

    def strict_gbk_clean(text):
        """严格清理，确保GBK兼容"""
        if pd.isna(text):
            return text

        text_str = str(text)

        try:
            # 步骤1: 修复编码
            fixed = text_str.encode('latin-1').decode('gb18030', errors='replace')

            # 步骤2: 移除GBK不支持的字符
            # GBK编码范围: 第一个字节0x81-0xFE, 第二个字节0x40-0x7E或0x80-0xFE
            cleaned = ''
            i = 0
            while i < len(fixed):
                char = fixed[i]
                try:
                    # 尝试编码为GBK
                    char.encode('gbk')
                    cleaned += char
                    i += 1
                except UnicodeEncodeError:
                    # 无法编码的字符替换为问号
                    cleaned += '?'
                    i += 1

            return cleaned

        except Exception:
            return text_str

    # 处理文本列
    text_cols = df.select_dtypes(include=['object']).columns

    for col in text_cols:
        print(f"严格处理: {col}")
        df[col] = df[col].apply(strict_gbk_clean)

    # 保存为GBK
    output_file = 'Data2015_gbk_strict.csv'
    df.to_csv(output_file, encoding='gbk', index=False)
    print(f"严格GBK转换完成: {output_file}")

    # 验证
    validate_gbk_file(output_file)


def validate_gbk_file(file_path):
    """验证GBK文件是否可读"""
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        print(f"GBK验证成功: {len(df)} 行数据")
        return True
    except Exception as e:
        print(f"GBK验证失败: {e}")
        return False


if __name__ == "__main__":
    strict_gbk_conversion()