# 数据导入
在电信数据的根目录下，执行del.py，将删除header后的csv文件写入到csv目录下.
```
python del.py
```
将写好的header目录和import.sh文件复制到电信数据的文件目录下,执行脚本import.sh.
执行前需正确设置环境变量NEO4J_HOME
```
./import.sh
```
