- 首先运行 python ./normalization.py -i ./dataset/sard 对数据进行处理，主要目的是将原来的数据集进行优化，优化的点包括以下：
1. 删除注释
2. 规范化参数和函数名（将参数名变为var，函数名变为fun）

- 生成 .bin文件  使用的是joern 1.1.1125
python joern_graph_gen.py  -i ../data/sard/Vul -o ../data/sard/bins/Vul -t parse
python joern_graph_gen.py  -i ../data/sard/No-Vul -o ../data/sard/bins/No-Vul -t parse  # 注意将 --out 改为--output


# 要注意修改 -r（ast,cfg,pdg和 cpg14）
# -o 的路径可以换(asts,cfgs,pdgs和 cpgs)
python joern_graph_gen.py -i ../data/sard/bins/Vul/ -o ../data/sard/pdgs/Vul/ -t export -r pdg
python joern_graph_gen.py -i ../data/sard/bins/No-Vul/ -o ../data/sard/pdgs/No-Vul/ -t export -r pdg  

Glove 文件夹,主要完成code embedding
- dot2vec.py dot文件转化为可以输入到glove的格式, 生成dotGlove.txt
- vectors.txt 经过 `demo.sh` 生成的关于源代码的embedding 向量表 # 可以在demo.sh里面修改
- dotvectors128.txt 是dot生成的向量表
- 运行`code2vec_glove`将 `dotvectors128.txt` 变成可以后面直接使用的模型 `sard_Vul_glove`


`pdgs`和`pdg`可换
python dot2csv.py -i ./data/sard/`pdgs`/Vul -o ./data/sard/csv/pdgcsv -r `pdg`


- deletecpg_or_folder_in_cfg.py是用于删除原本构建cpg时出现错误和cfg中有部分存在文件夹的脚本

- 嵌入操作，根据得到的glove的word embedding来嵌入，构图

使用csv2dot.py 构图 注意修改里面的 cpg和 cpg-data.pt