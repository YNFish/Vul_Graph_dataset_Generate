import os
import argparse
import shutil

def parse_options():
    parser = argparse.ArgumentParser(description='delete Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/sard/cpgs/Vul')
    parser.add_argument('-t', '--type', help='If deleting a folder (f), process the cpg folder (cpg)', type=str, default='cpg')
    args = parser.parse_args()
    return args

def delete_cpg(outdir):
    if outdir[-1] == '/':
        outdir = outdir
    else:
        outdir += '/'
        
    for item in os.listdir(outdir):
        name = item
        out = os.path.join(outdir, name)
    
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("1-cpg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv "+file_path+' '+out+'.dot')
                    os.system("rm -rf "+out)
                    break
        except:
            pass

def delete_folder(outdir):
    items = os.listdir(outdir)
    
    # 遍历每个文件和文件夹
    for item in items:
        item_path = os.path.join(outdir, item)
        # 判断是否是文件夹
        if os.path.isdir(item_path):
            try:
            # 使用 shutil.rmtree() 函数删除整个文件夹及其内容
                shutil.rmtree(item_path)
                print(f"文件夹 '{item_path}' 已成功删除。")
            except OSError as e:
                print(f"删除文件夹 '{item_path}' 失败：{e}")
        


def main():
    args = parse_options()
    outdir = args.input
    type = args.type
    
    if type=='cpg':
        delete_cpg(outdir)
    elif type == 'f':
        delete_folder(outdir)
    

if __name__ == '__main__':
    main()