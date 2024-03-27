import os
import glob
import argparse
from collections import Counter


my_list = []

def parse_options():
    parser = argparse.ArgumentParser(description='getinfomation.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/sard/Vul/')
    # parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='../testdata/sard/bins/Vul')
    # parser.add_argument('-t', '--type', help='The type of procedures: parse or export', type=str, default='export')
    # parser.add_argument('-r', '--repr', help='The type of representation: pdg or lineinfo_json', type=str, default='pdg')
    args = parser.parse_args()
    return args


def getinfomation(input_floder):
    file_names = glob.glob(input_floder+"*.c")
    # print(file_names)
    for file_name in file_names:
        file_name = os.path.basename(file_name)
        # print(file_name)
        # result = file_name.split('_')[3]
        my_list.append(file_name.split('_')[3])
        
    counts = Counter(my_list)
    for item, count in counts.items():
        # print(f"元素 '{item}' 出现了 {count} 次")
        print(item)
    # print(len(my_list))
    
'''
CWE78
CWE126
CWE122
CWE590
CWE121
CWE195
CWE690
CWE124
CWE134
CWE194
CWE127
CWE197
'''
    
    
    
'''
元素 'CWE78' 出现了 963 次    不安全的命令行参数使用
元素 'CWE126' 出现了 682 次   缓冲区溢出读取 
元素 'CWE122' 出现了 2122 次  堆栈溢出
元素 'CWE590' 出现了 15 次    释放已释放的内存  不要
元素 'CWE121' 出现了 3560 次  栈溢出
元素 'CWE195' 出现了 468 次   数值计算错误
元素 'CWE690' 出现了 15 次    异常处理错误   不要
元素 'CWE124' 出现了 1230 次  边界内溢出
元素 'CWE134' 出现了 2208 次  输入验证和表示错误
元素 'CWE194' 出现了 24 次    数值计算错误    不要
元素 'CWE127' 出现了 1015 次  缓冲区下溢
元素 'CWE197' 出现了 1 次     数值计算错误   不要
12303

'''
    
    
def main():
    args = parse_options()
    input_path = args.input
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'
    
    getinfomation(input_floder=input_path)
    
    
if __name__ == "__main__":
    main()