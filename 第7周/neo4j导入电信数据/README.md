# 电信数据导入neo4j
## 导入步骤
1. 将数据解压到2080服务器`/home/d1/lyl/telecom2`目录下，并将header目录、import.sh文件和del.py文件复制到该目录下。
    - header目录：包含重新编写的csv文件的首行
    - import.sh：包含neo4j导入命令
    - del.py：删除csv文件的首行
2. 在`/home/d1/lyl/telecom2`目录下，执行`del.py`，脚本将删除.csv文件的首行，并重写入到`/csv`子目录下.
```
python del.py
```
3. 执行`import.sh`，脚本将加载数据到本地neo4j的数据库`testnode.db`中，执行前需正确设置环境变量`NEO4J_HOME`
```
./import.sh
```
## 导入结果
![QQ截图20221002112556](https://user-images.githubusercontent.com/56336922/193436684-01d5c4ec-04ec-420d-ba15-ccc3348f79ed.png)

上图显示已将数据导入到2080机器上，导入耗时21m 20s，包含2.46亿节点、3.97亿关系和10.92亿属性，峰值内存使用3.756GiB。

![QQ截图20221002112829](https://user-images.githubusercontent.com/56336922/193436691-0bcc9a09-4d44-4e85-bf38-697f1c306c1a.png)

启动neo4j服务，并进入数据库`testnode.db`中即可查看导入的图数据。
