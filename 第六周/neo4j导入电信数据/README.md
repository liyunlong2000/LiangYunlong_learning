# 数据导入
## 导入步骤
在电信数据的根目录下，执行`del.py`，将删除header后的.csv文件写入到csv目录下.
```
python del.py
```
将写好的header目录和import.sh文件复制到电信数据的文件目录下,执行脚本`import.sh`.
执行前需正确设置环境变量NEO4J_HOME
```
./import.sh
```
## 导入结果
![QQ截图20221002112556](https://user-images.githubusercontent.com/56336922/193436684-01d5c4ec-04ec-420d-ba15-ccc3348f79ed.png)


![QQ截图20221002112829](https://user-images.githubusercontent.com/56336922/193436691-0bcc9a09-4d44-4e85-bf38-697f1c306c1a.png)
